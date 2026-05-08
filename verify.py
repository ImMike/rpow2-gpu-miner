#!/usr/bin/env python3
"""Standalone verifier for an rpow2 (prefix, nonce) pair.

Useful for spot-checking that a solution emitted by any miner — this one,
the on-site browser worker, or a future custom kernel — actually meets the
rpow2 PoW spec.

Usage:
    python verify.py PREFIX_HEX NONCE DIFFICULTY_BITS

Example:
    python verify.py deadbeefcafef00d11223344556677ee 540224 16
"""

import hashlib
import sys


def trailing_zero_bits(d: bytes) -> int:
    """Match the rpow2 worker's trailing-zero-bit count: scan bytes
    from the end (byte[31]) toward the start, low-order bits per byte."""
    z = 0
    for b in reversed(d):
        if b == 0:
            z += 8
            continue
        c = 0
        while not (b & (1 << c)):
            c += 1
        return z + c
    return z


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    prefix_hex, nonce_str, bits_str = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        prefix = bytes.fromhex(prefix_hex)
        nonce  = int(nonce_str)
        bits   = int(bits_str)
    except ValueError as e:
        sys.exit(f"bad args: {e}")
    preimage = prefix + nonce.to_bytes(8, "little")
    digest = hashlib.sha256(preimage).digest()
    z = trailing_zero_bits(digest)
    print(f"preimage    {preimage.hex()}")
    print(f"sha256      {digest.hex()}")
    print(f"trailing 0s {z}  (target >= {bits})")
    sys.exit(0 if z >= bits else 1)


if __name__ == "__main__":
    main()
