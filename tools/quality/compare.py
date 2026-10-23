"""Compare teacher-forced log-probabilities from logit_dump runs against a reference run.

For every other run: mean KL(ref || run) per position, top-1 agreement with the reference, and nats/token of the
teacher text (lower = the run predicts the real answer better). The reference's own nats/token is shown too.

Usage: python3 tools/quality/compare.py data/quality/gpu data/quality/split data/quality/static
"""
import sys

import numpy as np

V = 150144


def load(prefix):
    lp = np.memmap(prefix + ".lp", dtype=np.float32, mode="r").reshape(-1, V)
    tok = np.fromfile(prefix + ".tok", dtype=np.int32)
    return lp, tok


def main():
    ref_name, *others = sys.argv[1:]
    ref, tok = load(ref_name)
    nll_ref = -ref[np.arange(len(tok)), tok]
    print(f"{'run':24} {'KL vs ref (nats)':>18} {'top-1 agree':>12} {'nats/token':>11}")
    print(f"{ref_name:24} {'-':>18} {'-':>12} {nll_ref.mean():11.4f}")
    for name in others:
        lp, tok2 = load(name)
        assert (tok2 == tok).all()
        kl = np.empty(len(tok))
        agree = np.empty(len(tok), bool)
        for i in range(0, len(tok), 64):  # chunks keep memory low
            r = ref[i:i + 64].astype(np.float64)
            x = lp[i:i + 64].astype(np.float64)
            kl[i:i + 64] = (np.exp(r) * (r - x)).sum(axis=1)
            agree[i:i + 64] = r.argmax(axis=1) == x.argmax(axis=1)
        nll = -lp[np.arange(len(tok)), tok]
        print(f"{name:24} {kl.mean():10.5f} (p99 {np.percentile(kl, 99):.4f}) {agree.mean():11.2%} {nll.mean():11.4f}")


if __name__ == "__main__":
    main()
