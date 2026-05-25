# Safetensors Header Padding Bug — `ruvnet/wifi-densepose-pretrained`

**Status:** Open. Affects the `model.safetensors` file currently published at
[`huggingface.co/ruvnet/wifi-densepose-pretrained`](https://huggingface.co/ruvnet/wifi-densepose-pretrained).
Workaround available — see [Workaround](#workaround) below.

## TL;DR

The header in our published `model.safetensors` is padded to an 8-byte boundary
with literal `\x00` bytes instead of the `0x20` (space) padding the
[safetensors spec](https://github.com/huggingface/safetensors#format) requires.
Strict readers — including the Rust `safetensors` crate, Candle, and the Python
`safetensors.torch.load_file` helper that wraps the Rust binding — reject the
file with `SafetensorError: trailing characters at line 1 column 1462`. Lenient
readers (e.g. the hand-rolled parsers in `scripts/export-onnx.py` and the JS
`SafeTensorsReader` in `vendor/ruvector/.../export.js`) accept it because they
strip trailing NULs before `JSON.parse`.

## Byte-level evidence

Inspecting the file downloaded from the HF repo:

| Offset | Bytes | Meaning |
|--------|-------|---------|
| `0..8`     | `b8 05 00 00 00 00 00 00` | `u64 little-endian` declared header length = **1464** |
| `8..1469`  | `{"...":{...}}` (1461 JSON bytes) | The actual JSON header terminates at byte **1461** |
| `1469..1472` | `00 00 00` | **Three NUL bytes** padding the JSON up to the declared 1464 |
| `1472..EOF` | `...` | Tensor data section |

`1461 % 8 == 5`, so the writer pads 3 bytes to reach the next 8-byte boundary
(1464). The padding bytes are left as `\x00` because the writer zero-initializes
the buffer up front and never overwrites the padding zone.

## What the spec actually says

[https://github.com/huggingface/safetensors#format](https://github.com/huggingface/safetensors#format)

> 8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of
> the header.
>
> N bytes: a JSON UTF-8 string representing the header. The header data MUST
> begin with a `{` character (0x7B). The header data MAY be trailing padded with
> whitespace (0x20).

Whitespace = `0x20` (space). NUL (`0x00`) is not whitespace, and the strict
parsers correctly refuse to ignore it.

## Where the bug originates

The bad header is produced by `SafeTensorsWriter.build()` in
[`vendor/ruvector/npm/packages/ruvllm/src/export.js`](../../vendor/ruvector/npm/packages/ruvllm/src/export.js)
(part of the vendored `ruvnet/ruvector` submodule, source at
[https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)),
specifically lines 95-105:

```js
// Pad header to 8-byte alignment
const headerPadding = (8 - (headerBytes.length % 8)) % 8;
const paddedHeaderLength = headerBytes.length + headerPadding;
// ...
const totalLength = 8 + paddedHeaderLength + offset;
const buffer = new Uint8Array(totalLength);          // zero-initialised
const view = new DataView(buffer.buffer);
view.setBigUint64(0, BigInt(paddedHeaderLength), true);
buffer.set(headerBytes, 8);                          // padding zone untouched
```

`new Uint8Array(totalLength)` zero-fills the buffer, then only the JSON bytes
are copied in. The padding region between `headerBytes.length` and
`paddedHeaderLength` is never overwritten, so it stays `\x00`.

The corresponding `SafeTensorsReader.parseHeader()` in the same file masks the
bug by stripping trailing NULs (`headerJson.replace(/\0+$/, '')`) before
`JSON.parse` — round-tripping through the same writer/reader pair therefore
succeeds, and the bug only surfaces in third-party strict readers.

Three trainer scripts go through this exact code path:

- `scripts/train-wiflow.js` — `SafeTensorsWriter` → `model.safetensors` (line 933)
- `scripts/train-ruvllm.js` — same (line 1541)
- `scripts/train-camera-free.js` — same (line 2276)
- `scripts/train-wiflow-supervised.js` — same import (line 60)

The HF publisher (`scripts/publish-huggingface.py`) just uploads whatever files
sit in `dist/models/`; it does not generate or modify the `.safetensors` bytes,
so the fix is **not** in this repo's publishing script.

The Python writer used by `scripts/train-count.py::write_safetensors` (lines
128-167) produces `count_v1.safetensors` and is independent of the JS writer.
It writes the JSON header at exactly its UTF-8 byte length with no padding,
which is also spec-compliant (the spec allows no padding), so that writer is
**not** affected.

## Affected consumers

| Reader | Behaviour |
|--------|-----------|
| Rust `safetensors::SafeTensors::deserialize` (`safetensors 0.4.x` / `0.5.x` / `0.7.x`) | **Rejects** with `Error while deserializing header: invalid JSON in header: trailing characters at line 1 column 1462` |
| Candle (`candle_core::safetensors::load`, uses the Rust crate) | **Rejects** with the same error |
| Python `safetensors.torch.load_file` (wraps the Rust crate) | **Rejects** with `SafetensorError: trailing characters at line 1 column 1462` |
| Python `safetensors.safe_open` | **Rejects** with the same error |
| HuggingFace Hub safetensors metadata indexer | Marks the file as malformed in the repo's metadata view |
| `scripts/export-onnx.py::load_safetensors` (our hand-rolled reader) | **Accepts** — slices `f.read(header_len)` and `JSON.parse`s after Python silently tolerates trailing NULs in a `bytes`→`str` decode followed by `json.loads`. Strictly speaking this works only because the JSON tokenizer reaches end of input mid-payload; some interpreter versions raise here. |
| `SafeTensorsReader.parseHeader()` (JS, in the vendored ruvllm) | **Accepts** — strips trailing NULs explicitly |

## Repro

A 10-line script that reproduces the exact strict failure mode against a
synthetic file constructed the same way the buggy writer does:

```python
import json, struct, tempfile, os
from safetensors import safe_open

tensors = {"lora.A": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
           "lora.B": {"dtype": "F32", "shape": [4, 4], "data_offsets": [64, 128]}}
hdr = json.dumps(tensors).encode("utf-8")
pad = (8 - len(hdr) % 8) % 8                 # mimic the JS writer
buf = bytearray(8 + len(hdr) + pad + 128)    # zero-initialised, like new Uint8Array(...)
buf[0:8] = struct.pack("<Q", len(hdr) + pad) # declared length includes the padding
buf[8:8 + len(hdr)] = hdr                    # JSON only; padding zone stays \x00
fd, p = tempfile.mkstemp(suffix=".safetensors"); os.write(fd, bytes(buf)); os.close(fd)
with safe_open(p, framework="numpy") as f:   # raises SafetensorError
    print(list(f.keys()))
```

Running this against `safetensors==0.7.0` prints:

```
SafetensorError: Error while deserializing header: invalid JSON in header:
trailing characters at line 1 column 143
```

(143, not 1462, because this header is shorter than the published file's; the
**class** of error is identical, and `1461 + 1` likewise lands at column 1462
on the real artifact.)

## Proposed upstream fix

In `vendor/ruvector/npm/packages/ruvllm/src/export.js`, the writer must
either:

**Option A — spec-correct padding (preferred):** fill the padding zone with
`0x20` instead of leaving it `\x00`:

```js
const buffer = new Uint8Array(totalLength);
buffer.fill(0x20, 8 + headerBytes.length, 8 + paddedHeaderLength); // pad with spaces
const view = new DataView(buffer.buffer);
view.setBigUint64(0, BigInt(paddedHeaderLength), true);
buffer.set(headerBytes, 8);
```

**Option B — no padding:** size the declared header to the exact JSON length and
drop the alignment step. The spec doesn't require alignment; the implicit
goal of the 8-byte align is so the tensor payload that follows is naturally
aligned, but the Rust reference reader handles unaligned payloads fine.

The corresponding `SafeTensorsReader.parseHeader()` can stop stripping NULs
once writers are fixed (it remains safe to keep it as a backwards-compat
guard for already-published artifacts).

A drive-by patch would live in `ruvnet/ruvector` (not in this repo). Once
the upstream fix lands and the submodule is bumped, the model needs to be
**re-trained or re-exported and re-uploaded** to HuggingFace — there is no way
to fix the published artifact in place from the writer side, only from the
file side (see workaround below).

## Workaround

A small utility ships at [`scripts/fix-safetensors-header.py`](../../scripts/fix-safetensors-header.py)
that loads any `.safetensors` file, detects `\x00` padding in the header
region, and rewrites it in-place with `0x20` (space) padding — preserving the
declared header length and every tensor byte, so the SHA-256 of the **tensor
data** is unchanged. Only the header padding bytes flip from NUL to space.

Usage:

```bash
# Download the broken file
huggingface-cli download ruvnet/wifi-densepose-pretrained \
    model.safetensors --local-dir models/wifi-densepose-pretrained

# Fix it in place
python scripts/fix-safetensors-header.py \
    models/wifi-densepose-pretrained/model.safetensors

# Load with strict tooling
python -c "
from safetensors.torch import load_file
state = load_file('models/wifi-densepose-pretrained/model.safetensors')
print({k: tuple(v.shape) for k, v in state.items()})
"
```

The utility is idempotent: a fixed file with no `\x00` padding bytes in the
header zone reports `already clean` and exits 0 without rewriting.

## Follow-ups

- [ ] Patch the upstream writer in
      [`ruvnet/ruvector`](https://github.com/ruvnet/ruvector) (Option A above).
- [ ] Bump the `vendor/ruvector` submodule once the upstream fix lands.
- [ ] Re-train (or re-export) `model.safetensors` with the fixed writer and
      re-upload to `ruvnet/wifi-densepose-pretrained`. The HuggingFace LFS
      pointer should change; consumers who pinned by `revision=` will keep
      pulling the broken file until they update.
- [ ] Add a release-time check (`scripts/publish-huggingface.py`) that opens
      every `.safetensors` file in `dist/models/` with the strict Python loader
      and aborts the upload on rejection — prevents future regressions.
- [ ] Remove the `headerJson.replace(/\0+$/, '')` workaround from
      `SafeTensorsReader.parseHeader()` once no published artifacts depend on
      it (lenient readers mask the bug for round-trip tests inside the
      training pipeline).
