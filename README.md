# rpow2 GPU Miner

A fast, cross-platform GPU miner for [**rpow2**](https://rpow2.com) — the modern tribute to Hal Finney's original [Reusable Proofs of Work](https://en.wikipedia.org/wiki/Hashcash#Reusable_proofs_of_work). Compiles a Vulkan compute shader on the fly and mines on **any modern GPU** (AMD, NVIDIA, Intel — discrete *or* integrated). No CUDA toolkit, no ROCm install, no driver fiddling.

```
$ python rpow2_gpu_miner.py
signed in: you@example.com  balance=42  minted=42
compiling SPIR-V kernel (first launch only)...
kernel ready.

minted #1     bits=25  solve=  29ms  attempts=    33,554,432  token=8a2c…
minted #2     bits=25  solve=  41ms  attempts=    50,331,648  token=1e93…
minted #3     bits=25  solve=  18ms  attempts=    16,777,216  token=c044…
…
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/) [![GPU: Vulkan](https://img.shields.io/badge/GPU-Vulkan-red.svg)](https://www.vulkan.org/) [![rpow2](https://img.shields.io/badge/network-rpow2.com-black.svg)](https://rpow2.com)

---

## Why GPU mine rpow2?

The on-site browser miner uses a wasm SHA-256 implementation pinned to a single CPU thread. That's perfect for the in-browser experience, but it leaves three to four orders of magnitude of throughput on the floor on any machine with a real GPU.

| Miner | Typical throughput | Time per token at 25 bits |
| --- | --- | --- |
| Browser tab (`hash-wasm` SHA-256, single thread) | ~1 MH/s | ~33 s |
| Modest desktop CPU (e.g. Ryzen 5, OpenSSL SHA-NI) | ~150 MH/s | ~0.2 s |
| **This miner**, integrated GPU (e.g. Iris Xe, Radeon 780M) | **~300 MH/s** | **~0.1 s** |
| **This miner**, mid-range discrete GPU (e.g. RTX 3060, RX 6600) | **~1–2 GH/s** | **~16–33 ms** |
| **This miner**, high-end GPU (e.g. RTX 4080, RX 7900) | **~3–5 GH/s** | **~7–11 ms** |

Numbers are wall-clock against the live `api.rpow2.com` endpoint, so they include HTTP round-trip on top of the kernel.

---

## Features

- **One-file Python miner.** Just `pip install` and run.
- **Real GPU mining** via Vulkan compute. No CUDA SDK, no ROCm, no shader compiler in the path — the kernel is JIT-compiled to SPIR-V the first time it runs.
- **Cross-vendor:** AMD (RDNA1+), NVIDIA (Maxwell+), Intel (Gen9+), Apple (M-series with MoltenVK), even most integrated GPUs and the Steam Deck.
- **Zero protocol divergence.** Bit-identical to the on-site browser worker — every solution is verified on the CPU with `hashlib` before submission.
- **Stateless and minimal.** Single Python file (~250 lines), stdlib HTTP, two pip deps (`taichi`, `numpy`).
- **Safe by default.** Won't run without an explicit cookie, prints what account it's mining for, supports `--rounds N` to stop after a fixed number of mints, graceful summary on `Ctrl-C`.

---

## Hardware support

| GPU vendor | Linux | Windows | macOS |
| --- | --- | --- | --- |
| AMD (RDNA, RDNA2, RDNA3) | ✓ via Mesa `radv` | ✓ via AMD driver | n/a |
| NVIDIA (Maxwell+) | ✓ proprietary or open driver | ✓ | n/a |
| Intel (Gen9+ / Arc) | ✓ via `anv` (Mesa) | ✓ | n/a |
| Apple Silicon | n/a | n/a | ✓ via MoltenVK |
| Integrated graphics (Vega, Iris, Radeon 700M/800M, etc.) | ✓ | ✓ | ✓ |

If `vulkaninfo` lists your GPU, this miner will run on it.

---

## Install

```bash
git clone https://github.com/ImMike/rpow2-gpu-miner.git
cd rpow2-gpu-miner

python -m venv .venv
source .venv/bin/activate          # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

That's it. Taichi ships its own SPIR-V toolchain inside its wheel — you don't need `glslc`, `shaderc`, or any vendor SDK.

### Linux note

You'll need a Vulkan loader and a vendor ICD installed. On Ubuntu/Debian:

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers     # AMD/Intel
# or, for NVIDIA, the proprietary driver already ships its own ICD
```

Verify with `ls /usr/share/vulkan/icd.d/` — you should see at least one `*_icd.*.json`.

---

## Get your session cookie

The miner authenticates the same way your browser does — by carrying the `rpow_session` cookie. To grab yours:

1. Sign in to [rpow2.com](https://rpow2.com) with your email (you'll get a magic link).
2. Open DevTools (F12 in most browsers) and switch to the **Network** tab.
3. Click the **MINE** button on the site once. You'll see a `POST` request to `api.rpow2.com/challenge`.
4. Click that request → **Headers** → **Request Headers** → copy the entire `cookie:` value. It will start with `rpow_session=…`.

Then expose it to the miner:

```bash
export RPOW_COOKIE='rpow_session=eyJ…paste-the-whole-thing…'
```

> Treat this cookie like a password. It identifies your account to the network until it expires.

---

## Quick start

```bash
# Mine continuously until you Ctrl-C:
python rpow2_gpu_miner.py

# Mine exactly 100 tokens and exit:
python rpow2_gpu_miner.py --rounds 100

# Quiet mode (only the summary at the end):
python rpow2_gpu_miner.py --rounds 100 --quiet
```

`--help` lists every flag.

---

## How it works

rpow2's proof-of-work is a single-block SHA-256 with a "trailing zero bits" target:

```
preimage  = nonce_prefix_bytes || little_endian_uint64(nonce)
digest    = SHA256(preimage)
accept    iff trailing_zero_bits(digest) >= difficulty_bits
```

`nonce_prefix` is a 16-byte value the server hands out per challenge, and `difficulty_bits` is the current network target (you'll see it in every `/challenge` response). Each token mint is an independent puzzle. The full mining flow is:

```
client                                  api.rpow2.com
  |  POST /challenge                       |
  | -------------------------------------> |
  |  { challenge_id, nonce_prefix, bits }  |
  | <------------------------------------- |
  |                                        |
  | (search for nonce on GPU)              |
  |                                        |
  |  POST /mint { challenge_id, nonce }    |
  | -------------------------------------> |
  |  { token: { id, value, issued_at } }   |
  | <------------------------------------- |
```

This miner expresses the same SHA-256 round function in Taichi's Python-with-decorators kernel language. At import time, Taichi JIT-compiles it into a SPIR-V compute shader and dispatches it through Vulkan. Each kernel launch fans out to ~67 million attempts (1,048,576 threads × 64 nonces per thread by default — tune with `--threads` and `--iters`).

Every nonce the kernel marks as a candidate is independently re-hashed on the CPU with Python's `hashlib.sha256` before the miner posts to `/mint`. That means a kernel bug (or a flaky GPU memory module) results in a clean failure, never an invalid submission.

---

## Performance tuning

The defaults work well from integrated graphics up through high-end discrete cards. If you want to squeeze the last few percent:

- **More threads, same iters.** `--threads 4194304 --iters 16` keeps total work the same but flattens the parallel dimension. Helps on cards with high SM/CU count.
- **More iters, same threads.** `--iters 256` reduces per-launch dispatch overhead. Helps on integrated GPUs where launch latency dominates.
- **Power efficiency vs latency.** Lower `--threads` reduces tail latency on each PoW; higher `--threads` reduces the number of kernel launches per token at the cost of "wasted" work after a candidate is found.

A reasonable sweep:

```bash
for t in 262144 1048576 4194304; do
  for i in 16 64 256; do
    python rpow2_gpu_miner.py --rounds 5 --threads $t --iters $i --quiet
  done
done
```

---

## Verifying a solution

`verify.py` is a zero-dependency CPU verifier (just `hashlib`) for any rpow2 PoW solution. Useful for cross-checking another miner's output, or debugging your own kernel.

```bash
python verify.py <prefix_hex> <nonce_decimal> <difficulty_bits>
```

Example:

```bash
$ python verify.py deadbeefcafef00d11223344556677ee 540224 16
preimage    deadbeefcafef00d11223344556677ee403e080000000000
sha256      1d1770fc5adb902a9c6da3dd4a6ba6a3e96961e8306cdd8cd57ff500f43d0000
trailing 0s 16  (target >= 16)
```

Exit code 0 if the solution meets the target, 1 otherwise.

---

## FAQ

**Q: Do I need root / admin?**
No. Vulkan ICDs are usually pre-installed on every desktop OS. You only need elevated privileges if you have to install Mesa or a vendor driver.

**Q: Will this work on a laptop with only integrated graphics?**
Yes. Modern integrated GPUs (Intel Iris Xe, AMD Radeon 780M / 880M, Apple M-series) typically deliver 200–500 MH/s on this kernel — comfortably faster than even a fast CPU on the in-browser miner.

**Q: It says "WebAssembly is not supported in this environment". Help?**
That's the in-browser miner's error, not this one. This Python miner doesn't use WebAssembly.

**Q: Can I run multiple instances on the same machine?**
Yes, though one process per GPU is enough — the kernel already saturates the device. If you have multiple GPUs and want to use them all, Taichi will pick one based on the order Vulkan enumerates them; set `VK_ICD_FILENAMES` to point at a single ICD JSON to pin a specific device.

**Q: Can I mine multiple accounts in parallel?**
Each instance binds to one cookie. Run as many instances as you have cookies, but please be a good citizen and don't sybil-flood the network — it makes mining harder for everyone (including you) by raising effective difficulty.

**Q: Does this work with the rpow2 testnet / a fork?**
Yes — set `RPOW_API_BASE` and `RPOW_ORIGIN` to the URLs of your fork before running. Both default to the public network.

**Q: Why Vulkan and not OpenCL / CUDA / Metal?**
Vulkan compute runs everywhere with one set of code. OpenCL has uneven driver support on consumer hardware, CUDA is NVIDIA-only, Metal is Apple-only. Taichi targets all three, and Vulkan is the lowest-friction choice for "works on any modern GPU."

---

## Troubleshooting

**`RuntimeError: [Taichi] No Vulkan device found`**
Your Vulkan loader isn't seeing any GPU. Check `vulkaninfo --summary` (Linux/Windows) or `system_profiler SPDisplaysDataType` (macOS). On Linux, install `mesa-vulkan-drivers` for AMD/Intel or the proprietary NVIDIA driver. Make sure `/usr/share/vulkan/icd.d/` has at least one ICD JSON.

**`auth check failed: http 401`**
Cookie expired or invalid. Re-grab it from your browser (see [Get your session cookie](#get-your-session-cookie)).

**`/challenge` is slow (10+ seconds per request)**
The network is rate-limiting your account or is genuinely overloaded. Both the rate-limit and the overload are server-side and there's nothing the client can do about them — just wait it out, or mine slower.

**Kernel JIT compile takes a long time (5–10 s)**
Normal on the first launch only. Taichi caches the compiled SPIR-V in `~/.cache/taichi/`; subsequent runs start in ~1 s.

---

## Related

- **rpow2** — the network: <https://rpow2.com>
- **Hal Finney's original RPOW** — the inspiration: <https://nakamotoinstitute.org/rpow/>
- **Taichi** — the GPU compute framework: <https://www.taichi-lang.org>
- **hash-wasm** — the in-browser SHA-256 implementation rpow2 ships: <https://github.com/Daninet/hash-wasm>

---

## Contributing

PRs welcome. A few low-hanging fruit:

- A native CUDA kernel for NVIDIA cards (typically 2–3× the Vulkan path).
- A native ROCm/HIP kernel for AMD cards (similar gains).
- A Web Worker version that does what the on-site miner does but on the browser's `WebGPU` (where available) instead of WASM SHA-256.
- A `--multi-gpu` flag that round-robins challenges across all enumerated devices.

Please run `python verify.py` on a handful of solutions before opening a PR that touches the kernel — that's the easy guardrail against subtle SHA-256 bugs.

---

## License

MIT — see [LICENSE](LICENSE).

This is community software for an open mining network. It's not affiliated with the operators of rpow2.com.
