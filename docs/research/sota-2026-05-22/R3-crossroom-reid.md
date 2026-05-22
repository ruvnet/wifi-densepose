# R3 — Cross-room CSI re-identification: AETHER + MERIDIAN synthesis

**Status:** simulation + ADR-024/027 synthesis + privacy framing · **2026-05-22**

## The question

AETHER (ADR-024) gives us contrastive CSI embeddings that achieve **~95% within-room 1-shot re-identification** on MM-Fi. Can the same embeddings identify the same person across a different room?

This question has two answers — a technical one and an ethical one. R3 takes both seriously.

## Decomposition

A CSI embedding from any frame is approximately:

```
embedding = person_signature + environment_signature + noise
```

The environment signature includes multipath geometry, AP placement, furniture, walls. It is **constant per (room, antenna placement)**, and **changes by O(1)** between rooms — empirically larger than the per-person signature variation. This is exactly the structure that ADR-027 (MERIDIAN) targets.

`examples/research-sota/r3_crossroom_reid.py` simulates the problem with physics-realistic parameters: 10 subjects, 3 rooms, 128-dim embeddings, person-signature scale 0.35, environment scale 1.5 (env ≈ 4.7× person), noise 0.3.

## Results

| Configuration | 1-shot accuracy | Δ from baseline |
|---|---:|---|
| Within-room baseline | 100.0% | (matches AETHER ~95% target) |
| Cross-room, **raw cosine** K-NN | **70.0%** | -30 pp |
| Cross-room, MERIDIAN 100% env subtraction | 100.0% | recovered |
| Cross-room, MERIDIAN 70% env subtraction (realistic) | 100.0% | recovered |
| Chance | 10.0% | floor |

Three observations:

1. **Cosine K-NN partially mitigates** the environment-shift problem (70% >> 10% chance) because magnitude normalisation removes the additive env component as a *direction*. The remaining 30 pp gap comes from how the env shift rotates the cluster in the high-dim space.
2. **Explicit MERIDIAN-style env subtraction** (per-room centroid removal) closes the remaining gap. The simulation suggests even **70%-effective** subtraction (realistic for finite labelled examples) is enough.
3. **The within-room baseline is what an attacker has**, not what the system needs. The same primitive that gives the user "let RuView greet you by name in this room" also gives an attacker "this person walked through 5 different rooms and we tracked them."

## Why the env-removal approach works

MERIDIAN's core idea (ADR-027) is to estimate `environment_signature` from labelled samples *in the new room* and subtract it. The estimator works because:

- All people contribute equally to the per-room mean (assuming reasonably balanced training data)
- The person signatures are zero-mean across the population (an embedding is meaningful only relative to others)
- Therefore `mean(embeddings in room R) ≈ environment_signature[R]`

Subtracting the per-room centroid gives `embedding_clean ≈ person_signature + noise`, which is the room-invariant signature.

**Trade-off:** MERIDIAN needs labelled (or at least clustered) examples *in the new room* to estimate its centroid. Pure zero-shot transfer to an unobserved room is much harder — without any anchor, you can't distinguish "person A in new room" from "person B in old room" robustly.

## Physics gives us another lever

R6's Fresnel forward model tells us where the env_sig **lives** in the embedding: it's the contribution from the multipath / reflector geometry. A 5 m bedroom has 4-6 dominant reflector positions; the env_sig is a function of those.

If we could **predict** the env_sig from the forward model + a room geometry (R6's A matrix + a coarse map of the room), we wouldn't need labelled examples. This is the next-tier sophistication: **physics-informed domain invariance** rather than statistically estimated.

This isn't built. It's the right next step in the AETHER + MERIDIAN line.

## Privacy framing (the ethical answer)

The same primitive that enables "RuView greets you by name in your bedroom" enables a building-level adversary to **track every individual's movement through every WiFi-CSI-sensing surface**. This is a stronger surveillance primitive than face recognition because:

- WiFi penetrates walls (no line-of-sight needed)
- Re-ID works without subject cooperation (no "look at the camera")
- The signal is invisible (no light, no observable signal)
- The biometric is the body's RF signature, not a removable accessory

The R14 ethical framework (opt-in by default, data stays on-device, override is one tap) applies, but with **additional** constraints specific to re-ID:

1. **No cross-installation linkage.** Per-installation embedding spaces only. Two RuView installs in two different buildings must NOT share embedding spaces.
2. **Embedding storage requires explicit opt-in.** Storing person embeddings persists biometrics; many regulatory regimes treat this as biometric data with stronger consent requirements (GDPR Art 9, BIPA).
3. **Forgetting must be cryptographically verifiable.** When a user requests deletion, the embedding must be cryptographically destroyed, not just unlabelled. Storing "unlabelled embeddings" still enables future linkage.
4. **No re-ID across legal entities.** Building A and Building B owned by different entities must NOT exchange embeddings. The data-flow boundaries should be hard-walled.

These constraints make some use cases impossible (e.g. "automatic global biometric ID" — yes, that's the point) and some clearly aligned with the user (e.g. "remember which family member is in which room").

## What this enables

1. **Per-installation personalisation** — empathic appliances (R14) get per-person calibration after MERIDIAN-style env subtraction.
2. **Anomaly detection** — "someone walked into this room who isn't in the household's embedding set" → home-security primitive without face recognition.
3. **Pose-data-association** — multi-person pose tracking in the same room can use the embedding to maintain consistent identity through occlusion.

## What this DOES NOT enable (correctly, by design)

1. Cross-building tracking
2. Re-ID across legal entities
3. Long-term unlabelled biometric storage
4. Zero-shot transfer to unobserved rooms (without physics-informed extension)

## Honest scope

- The simulation uses additive `person + env + noise` decomposition. Real CSI has **multiplicative** environment effects in the multipath domain — env modulates person signature amplitude in subcarrier-specific ways. A more realistic forward model would multiply the per-subcarrier slot transfer function with the person signature, which makes env-removal harder (not just subtraction).
- The 70% cross-room raw cosine K-NN number depends heavily on env / person scale ratio. With a 10× larger env (e.g. crossing from a bedroom to a kitchen with very different multipath), the raw cosine K-NN drops further. With a 2× smaller env (very similar rooms), it barely drops. The MERIDIAN closing of the gap appears robust.
- We did **not** simulate adversarial scenarios where an attacker actively manipulates the env signal to break tracking. R7's mincut would have to weigh in on this.

## Connection back

- **R5** (saliency) — within-room saliency profiles include both the person- and environment-saliency. Cross-room transfer would need to find the *person-only* saliency, which is a research problem AETHER (ADR-024) partially addresses through contrastive learning.
- **R6** (Fresnel) — the missing piece: physics-informed env_sig prediction from a room model. Not yet built.
- **R7** (mincut adversarial) — cross-room re-ID is the highest-risk surface for adversarial spoofing. If the system can be fooled into thinking "person B is in room A", that's a security incident; multi-link consistency from R7 is the defence.
- **R9** (RSSI K-NN) — already showed that even RSSI alone preserves a weak locality signature within room; the cross-room transfer for RSSI is *worse* than for full CSI, but the env / person decomposition still applies.
- **R14** (empathic appliances) — re-ID enables per-occupant V1 lighting / V2 HVAC / V3 attention-respecting. The privacy constraints from R14 + the four cross-installation constraints from R3 together are the binding spec.

## Next ticks (R3 follow-ups)

- Physics-informed env_sig prediction from R6's forward operator + a coarse room map → zero-shot cross-room transfer.
- Multi-occupant re-ID under occlusion: two people in the same room, intermittent visibility of each; can a Kalman + AETHER pipeline maintain identity continuously?
- Cryptographic forgetting protocol: how do you prove an embedding has been deleted to a regulator who can't see your hard drive? (Out of scope for this loop, but a real research question.)
