"""
Phase 5 — OccWorld VQVAE + Transformer retraining on RuView indoor occupancy.

Two-stage training pipeline:
  Stage 1: Retrain VQVAE tokenizer on RuView snapshots
  Stage 2: Retrain autoregressive transformer on tokenized sequences

Usage:
    # Stage 1: VQVAE
    python3 scripts/occworld_retrain.py vqvae \
        --snapshots /tmp/snapshots/ \
        --work-dir out/ruview_vqvae \
        --epochs 200

    # Stage 2: Transformer (requires Stage 1 checkpoint)
    python3 scripts/occworld_retrain.py transformer \
        --snapshots /tmp/snapshots/ \
        --vqvae-checkpoint out/ruview_vqvae/latest.pth \
        --work-dir out/ruview_occworld \
        --epochs 200

    # Generate training snapshots from the live sensing server
    python3 scripts/occworld_retrain.py record \
        --server http://localhost:8080 \
        --out-dir /tmp/snapshots/scene_live \
        --duration 3600

Requirements:
    ml-env with OccWorld installed (see ADR-147 §3)
    At least 16 GB VRAM for training (RTX 5080 sufficient at batch=1)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


# ── Stage 0: Record snapshots from the live sensing server ───────────────────

def cmd_record(args: argparse.Namespace) -> None:
    """Stream WorldGraph snapshots from the sensing server REST API."""
    import json
    import urllib.request

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{args.server.rstrip('/')}/api/v1/worldgraph/snapshot"
    end_time = time.time() + args.duration
    frame_idx = 0
    interval = args.interval

    log.info("Recording snapshots from %s → %s for %ds", url, out_dir, args.duration)

    while time.time() < end_time:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                snap = json.loads(resp.read())
            out_path = out_dir / f"frame_{frame_idx:06d}.json"
            out_path.write_text(json.dumps(snap))
            frame_idx += 1
            if frame_idx % 100 == 0:
                log.info("Recorded %d frames", frame_idx)
        except Exception as exc:
            log.warning("Snapshot fetch failed: %s", exc)
        time.sleep(interval)

    log.info("Done — recorded %d frames to %s", frame_idx, out_dir)


# ── Stage 1: VQVAE retraining ────────────────────────────────────────────────

def cmd_vqvae(args: argparse.Namespace) -> None:
    """Retrain the OccWorld VQVAE tokenizer on RuView indoor occupancy."""
    sys.path.insert(0, str(Path(args.occworld_dir).resolve()))

    import torch
    from mmengine.config import Config
    from mmengine.registry import MODELS

    try:
        import model as occmodel  # noqa: F401 — registers custom MODELS
    except ImportError:
        log.error("Could not import OccWorld model package. Set --occworld-dir correctly.")
        sys.exit(1)

    from ruview_occ_dataset import RuViewOccDataset

    cfg = Config.fromfile(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build VQVAE only
    vae = MODELS.build(cfg.model.vae).cuda()
    log.info("VQVAE params: %.1fM", sum(p.numel() for p in vae.parameters()) / 1e6)

    ds = RuViewOccDataset(
        args.snapshots,
        return_len=cfg.model.get("num_frames", 15) + 1,
        voxel_m=args.voxel_m,
        x_min=args.x_min,
        y_min=args.y_min,
    )
    log.info("Dataset: %d windows from %s", len(ds), args.snapshots)

    if len(ds) == 0:
        log.error("No training windows found in %s — record snapshots first.", args.snapshots)
        sys.exit(1)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=not args.no_shuffle, num_workers=0,
        collate_fn=lambda b: b[0],  # dict passthrough
    )

    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0.0
        for batch in loader:
            occ = torch.from_numpy(batch["target_occs"]).long().unsqueeze(0).cuda()  # (1,F,H,W,D)
            # VQVAE forward: encode + quantize + decode, returns reconstruction loss
            z, shape = vae.forward_encoder(occ)
            z = vae.vqvae.quant_conv(z)
            z_q, vq_loss, _ = vae.vqvae.forward_quantizer(z, is_voxel=False)
            z_q = vae.vqvae.post_quant_conv(z_q)
            recon = vae.forward_decoder(z_q, shape, occ.shape)
            recon_loss = torch.nn.functional.cross_entropy(
                recon.flatten(0, -2),
                occ.flatten(),
            )
            loss = recon_loss + vq_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        if epoch % 10 == 0:
            log.info("Epoch %d/%d  loss=%.4f  lr=%.2e", epoch + 1, args.epochs, avg, scheduler.get_last_lr()[0])

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "state_dict": vae.state_dict(), "loss": avg},
                       work_dir / "latest.pth")

    log.info("VQVAE training complete. Best loss=%.4f  checkpoint: %s/latest.pth",
             best_loss, work_dir)


# ── Stage 2: Transformer retraining ─────────────────────────────────────────

def cmd_transformer(args: argparse.Namespace) -> None:
    """Retrain the OccWorld autoregressive transformer on tokenized RuView sequences."""
    sys.path.insert(0, str(Path(args.occworld_dir).resolve()))

    import torch
    from copy import deepcopy
    from einops import rearrange
    from mmengine.config import Config
    from mmengine.registry import MODELS

    try:
        import model as occmodel  # noqa: F401
    except ImportError:
        log.error("OccWorld model package not found.")
        sys.exit(1)

    from ruview_occ_dataset import RuViewOccDataset

    cfg = Config.fromfile(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    full_model = MODELS.build(cfg.model).cuda()

    # Load VQVAE checkpoint if provided
    if args.vqvae_checkpoint:
        ck = torch.load(args.vqvae_checkpoint, map_location="cuda")
        full_model.vae.load_state_dict(ck["state_dict"])
        log.info("Loaded VQVAE checkpoint: %s", args.vqvae_checkpoint)
    full_model.vae.eval()
    for p in full_model.vae.parameters():
        p.requires_grad_(False)

    log.info("Transformer params: %.1fM",
             sum(p.numel() for p in full_model.transformer.parameters()) / 1e6)

    ds = RuViewOccDataset(args.snapshots, return_len=cfg.model.get("num_frames", 15) + 1)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=lambda b: b[0],
    )

    opt = torch.optim.AdamW(full_model.transformer.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        full_model.transformer.train()
        epoch_loss = 0.0
        for batch in loader:
            occ = torch.from_numpy(batch["target_occs"]).long().unsqueeze(0).cuda()
            with torch.no_grad():
                z, shape = full_model.vae.forward_encoder(occ)
                z = full_model.vae.vqvae.quant_conv(z)
                z_q, _, (_, _, indices) = full_model.vae.vqvae.forward_quantizer(z, is_voxel=False)
                z_q = rearrange(z_q, "(b f) c h w -> b f c h w", b=1)

            bs, F, C, H, W = z_q.shape
            pose_tokens = torch.zeros(bs, full_model.num_frames, C, device=z_q.device)
            pred_tokens, _ = full_model.transformer(z_q[:, :full_model.num_frames], pose_tokens)
            indices_target = rearrange(indices, "(b f) h w -> b f h w", b=bs)[:, full_model.offset:]
            loss = torch.nn.functional.cross_entropy(
                pred_tokens.flatten(0, 1),
                indices_target.flatten(0, 1).flatten(1),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.transformer.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        if epoch % 10 == 0:
            avg = epoch_loss / max(len(loader), 1)
            log.info("Epoch %d/%d  loss=%.4f", epoch + 1, args.epochs, avg)
            torch.save({"epoch": epoch, "state_dict": full_model.state_dict(), "loss": avg},
                       work_dir / "latest.pth")

    log.info("Transformer training complete. Checkpoint: %s/latest.pth", work_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OccWorld retraining pipeline for RuView (ADR-147 Phase 5)")
    p.add_argument("--occworld-dir", default=os.path.expanduser("~/projects/OccWorld"),
                   help="Path to OccWorld repo root")
    p.add_argument("--config", default=os.path.expanduser("~/projects/OccWorld/config/occworld.py"),
                   help="OccWorld config file")

    sub = p.add_subparsers(dest="cmd", required=True)

    # record
    rec = sub.add_parser("record", help="Record WorldGraph snapshots from sensing server")
    rec.add_argument("--server", default="http://localhost:8080")
    rec.add_argument("--out-dir", required=True)
    rec.add_argument("--duration", type=int, default=3600, help="Recording duration (s)")
    rec.add_argument("--interval", type=float, default=0.5, help="Poll interval (s)")

    # vqvae
    vae = sub.add_parser("vqvae", help="Retrain VQVAE tokenizer")
    vae.add_argument("--snapshots", required=True)
    vae.add_argument("--work-dir", default="out/ruview_vqvae")
    vae.add_argument("--epochs", type=int, default=200)
    vae.add_argument("--voxel-m", type=float, dest="voxel_m", default=0.4)
    vae.add_argument("--x-min", type=float, dest="x_min", default=-40.0)
    vae.add_argument("--y-min", type=float, dest="y_min", default=-40.0)
    vae.add_argument("--no-shuffle", action="store_true")

    # transformer
    xfm = sub.add_parser("transformer", help="Retrain autoregressive transformer")
    xfm.add_argument("--snapshots", required=True)
    xfm.add_argument("--vqvae-checkpoint", default=None)
    xfm.add_argument("--work-dir", default="out/ruview_occworld")
    xfm.add_argument("--epochs", type=int, default=200)

    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    {"record": cmd_record, "vqvae": cmd_vqvae, "transformer": cmd_transformer}[args.cmd](args)
