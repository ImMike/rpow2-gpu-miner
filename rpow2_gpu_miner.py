#!/usr/bin/env python3
"""rpow2 GPU miner.

Mines rpow2 SHA-256 proof-of-work on any Vulkan-capable GPU (AMD, NVIDIA,
Intel, integrated). Uses Taichi to JIT-compile a SPIR-V compute shader from
the Python source — no separate driver toolchain required.

Quick start:

    pip install -r requirements.txt
    export RPOW_COOKIE='rpow_session=...'   # paste from your browser DevTools
    python rpow2_gpu_miner.py

Run forever, or stop after N tokens with `--rounds N`. See `--help` for all
flags.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import taichi as ti

# --------------------------------------------------------------------------
# rpow2 API constants. Override via env if you're testing against a fork.
# --------------------------------------------------------------------------
API_BASE = os.environ.get("RPOW_API_BASE", "https://api.rpow2.com")
ORIGIN = os.environ.get("RPOW_ORIGIN", "https://rpow2.com")
USER_AGENT = "rpow2-gpu-miner/1.0 (+https://github.com/ImMike/rpow2-gpu-miner)"


# --------------------------------------------------------------------------
# HTTP helper. Stdlib only, no `requests` dependency.
# --------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"http {status}: {body}")
        self.status = status
        self.body = body


def http(method: str, path: str, cookie: str, body=None, timeout: float = 60.0):
    headers = {
        "cookie": cookie,
        "origin": ORIGIN,
        "user-agent": USER_AGENT,
        "accept": "application/json",
    }
    data = None
    if body is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        raise ApiError(e.code, parsed) from None


# --------------------------------------------------------------------------
# Taichi / Vulkan kernel.
#
# rpow2 PoW spec (matches the on-site browser worker):
#   preimage = nonce_prefix_bytes || little-endian uint64 nonce
#   accept iff trailing_zero_bits(SHA-256(preimage)) >= difficulty_bits
#
# `nonce_prefix` is 16 bytes (32 hex chars) issued by POST /challenge.
# Total preimage is 24 bytes; the SHA-256 padded message fits in one block.
# --------------------------------------------------------------------------
ti.init(arch=ti.vulkan, log_level=ti.WARN)

_K_TABLE = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
_H0_TABLE = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

K = ti.field(dtype=ti.u32, shape=64)
H0 = ti.field(dtype=ti.u32, shape=8)
for _i, _v in enumerate(_K_TABLE):
    K[_i] = _v
for _i, _v in enumerate(_H0_TABLE):
    H0[_i] = _v


@ti.func
def _rotr(x: ti.u32, n: ti.i32) -> ti.u32:
    return (x >> n) | (x << (32 - n))


@ti.kernel
def mine_kernel(
    n_threads: ti.i32,
    iters: ti.i32,
    base_nonce: ti.u32,
    p0: ti.u32, p1: ti.u32, p2: ti.u32, p3: ti.u32,
    bit_mask: ti.u32,
    result: ti.types.ndarray(dtype=ti.u32, ndim=1),  # [found, nonce, _pad]
):
    for gid in range(n_threads):
        local_base = base_nonce + ti.u32(gid) * ti.u32(iters)
        for k in range(iters):
            nonce = local_base + ti.u32(k)
            # SHA-256 reads each 4-byte word big-endian; nonce bytes are LE,
            # so W[4] is the byte-swap of nonce_lo.
            w4 = ((nonce & ti.u32(0xff)) << 24) \
                | (((nonce >> 8) & ti.u32(0xff)) << 16) \
                | (((nonce >> 16) & ti.u32(0xff)) << 8) \
                | ((nonce >> 24) & ti.u32(0xff))

            W = ti.Vector.zero(ti.u32, 64)
            W[0] = p0
            W[1] = p1
            W[2] = p2
            W[3] = p3
            W[4] = w4
            W[5] = ti.u32(0)
            W[6] = ti.u32(0x80000000)  # SHA-256 padding marker
            W[15] = ti.u32(192)        # bit length: 24 bytes = 192 bits

            for t in range(16, 64):
                s0 = _rotr(W[t - 15], 7) ^ _rotr(W[t - 15], 18) ^ (W[t - 15] >> 3)
                s1 = _rotr(W[t - 2], 17) ^ _rotr(W[t - 2], 19) ^ (W[t - 2] >> 10)
                W[t] = W[t - 16] + s0 + W[t - 7] + s1

            a = H0[0]; b = H0[1]; c = H0[2]; d = H0[3]
            e = H0[4]; f = H0[5]; g = H0[6]; h = H0[7]

            for t in range(64):
                S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
                ch = (e & f) ^ ((~e) & g)
                temp1 = h + S1 + ch + K[t] + W[t]
                S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
                mj = (a & b) ^ (a & c) ^ (b & c)
                temp2 = S0 + mj
                h = g; g = f; f = e
                e = d + temp1
                d = c; c = b; b = a
                a = temp1 + temp2

            h7 = H0[7] + h
            # For target_bits in [1..32], "trailing zero bits >= target" iff
            # (h7 & ((1<<bits)-1)) == 0, because h7 is the last 4 digest bytes
            # and the rpow2 trailing-zero count starts at byte[31].
            if (h7 & bit_mask) == ti.u32(0):
                prev = ti.atomic_or(result[0], ti.u32(1))
                if prev == ti.u32(0):
                    result[1] = nonce


def _hex_prefix_to_uint32_be(prefix_hex: str):
    pb = bytes.fromhex(prefix_hex)
    if len(pb) != 16:
        raise ValueError(f"expected 16-byte prefix, got {len(pb)}")
    return tuple(int.from_bytes(pb[i:i + 4], "big") for i in range(0, 16, 4))


def _trailing_zero_bits(d: bytes) -> int:
    z = 0
    for byte in reversed(d):
        if byte == 0:
            z += 8
            continue
        c = 0
        while not (byte & (1 << c)):
            c += 1
        return z + c
    return z


def verify(prefix_hex: str, nonce: int, target_bits: int) -> bool:
    """Sanity-check that a (prefix, nonce) really meets the target."""
    msg = bytes.fromhex(prefix_hex) + nonce.to_bytes(8, "little")
    return _trailing_zero_bits(hashlib.sha256(msg).digest()) >= target_bits


def solve(prefix_hex, target_bits, n_threads, iters, attempt_cap):
    """Return (winning_nonce, total_attempts). Raises RuntimeError on cap."""
    if not (1 <= target_bits <= 31):
        raise ValueError(f"target_bits must be 1..31, got {target_bits}")
    p0, p1, p2, p3 = _hex_prefix_to_uint32_be(prefix_hex)
    bit_mask = np.uint32((1 << target_bits) - 1)
    base = np.uint32(0)
    total = 0
    while total < attempt_cap:
        result_buf = np.zeros(3, dtype=np.uint32)
        mine_kernel(
            n_threads, iters, int(base),
            np.uint32(p0), np.uint32(p1), np.uint32(p2), np.uint32(p3),
            bit_mask, result_buf,
        )
        ti.sync()
        total += n_threads * iters
        if int(result_buf[0]) == 1:
            nonce = int(result_buf[1])
            if not verify(prefix_hex, nonce, target_bits):
                raise RuntimeError(
                    f"kernel returned nonce={nonce} that does not verify on CPU; "
                    "this should never happen — please file a bug"
                )
            return nonce, total
        base = (base + np.uint32(n_threads * iters)) & np.uint32(0xFFFFFFFF)
    raise RuntimeError(f"attempt_cap reached after {total} hashes")


# --------------------------------------------------------------------------
# Live mining loop.
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="GPU-accelerated rpow2 miner (Vulkan/Taichi).",
    )
    p.add_argument(
        "--cookie",
        default=os.environ.get("RPOW_COOKIE"),
        help="rpow2 session cookie. Format: 'rpow_session=<value>'. "
             "Defaults to $RPOW_COOKIE.",
    )
    p.add_argument("--rounds", type=int, default=0,
                   help="stop after N successful mints (0 = run forever)")
    p.add_argument("--threads", type=int, default=1 << 20,
                   help="GPU threads per kernel launch (default: 1048576)")
    p.add_argument("--iters", type=int, default=64,
                   help="nonces per thread per launch (default: 64)")
    p.add_argument("--attempt-cap", type=int, default=1 << 36,
                   help="abort a single PoW after this many attempts (safety)")
    p.add_argument("--quiet", action="store_true",
                   help="only print summary, no per-mint lines")
    args = p.parse_args()

    if not args.cookie:
        sys.exit(
            "no cookie supplied. set $RPOW_COOKIE or pass --cookie. "
            "tip: in your browser, DevTools → Network → click any /api request "
            "→ copy the 'cookie' header (starts with 'rpow_session=')."
        )
    if not args.cookie.startswith("rpow_session="):
        sys.exit("cookie does not start with 'rpow_session='. paste the full "
                 "header value, not just the JWT-looking part.")

    # Confirm auth before warming up the GPU.
    try:
        status, me = http("GET", "/me", args.cookie)
    except ApiError as e:
        sys.exit(f"auth check failed: {e}")
    if status != 200 or not me or "email" not in me:
        sys.exit(f"unexpected /me response: {me}")
    print(f"signed in: {me['email']}  balance={me['balance']}  minted={me['minted']}",
          file=sys.stderr, flush=True)

    print("compiling SPIR-V kernel (first launch only)...", file=sys.stderr, flush=True)
    # Warm-up — never finds (mask=0xFFFFFFFF) but JIT-compiles the kernel.
    warm = np.zeros(3, dtype=np.uint32)
    mine_kernel(
        args.threads, args.iters, np.uint32(0),
        np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
        np.uint32(0xFFFFFFFF), warm,
    )
    ti.sync()
    print("kernel ready.\n", file=sys.stderr, flush=True)

    minted = 0
    failures = 0
    started_at = time.time()

    def stop_summary(*_):
        elapsed = time.time() - started_at
        print(file=sys.stderr)
        print("---- summary ----", file=sys.stderr)
        print(f"minted:    {minted}", file=sys.stderr)
        print(f"failures:  {failures}", file=sys.stderr)
        print(f"elapsed:   {elapsed:.1f}s", file=sys.stderr)
        if elapsed > 0:
            print(f"avg rate:  {minted/elapsed:.2f} tokens/sec  "
                  f"(~{minted/elapsed*3600:.0f}/hour)", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT,  stop_summary)
    signal.signal(signal.SIGTERM, stop_summary)

    while True:
        if args.rounds and minted >= args.rounds:
            break

        try:
            _, ch = http("POST", "/challenge", args.cookie)
        except ApiError as e:
            failures += 1
            print(f"[!] /challenge failed: {e}", file=sys.stderr, flush=True)
            time.sleep(1.0)
            continue

        cid    = ch["challenge_id"]
        prefix = ch["nonce_prefix"]
        bits   = ch["difficulty_bits"]

        t0 = time.time()
        try:
            nonce, attempts = solve(
                prefix, bits, args.threads, args.iters, args.attempt_cap,
            )
        except RuntimeError as e:
            failures += 1
            print(f"[!] solve failed for challenge {cid}: {e}",
                  file=sys.stderr, flush=True)
            continue
        solve_ms = (time.time() - t0) * 1000.0

        try:
            _, m = http(
                "POST", "/mint", args.cookie,
                {"challenge_id": cid, "solution_nonce": str(nonce)},
            )
        except ApiError as e:
            failures += 1
            print(f"[!] /mint failed (challenge {cid}): {e}",
                  file=sys.stderr, flush=True)
            continue

        minted += 1
        token_id = (m or {}).get("token", {}).get("id", "?")
        if not args.quiet:
            print(
                f"minted #{minted:<5d}  bits={bits}  solve={solve_ms:>5.0f}ms  "
                f"attempts={attempts:>10,}  token={token_id}",
                flush=True,
            )

    stop_summary()


if __name__ == "__main__":
    main()
