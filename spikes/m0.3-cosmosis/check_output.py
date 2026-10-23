"""Check a CosmoSIS text chain from the toy posterior against the analytic answer.

Checks:
1. every row carries derived values, and they equal ``toy.derived(params)``;
2. weighted posterior mean and standard deviations match the analytic Gaussian;
3. ln Z (if the sampler reports it) matches the analytic value.

Usage: python check_output.py path/to/chain.txt
"""

import sys

import numpy as np
import toy


def load(path):
    names, meta = None, {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if names is None:
                names = line[1:].split()
            elif "=" in line:
                k, v = line[1:].strip().split("=", 1)
                meta[k] = v.split()[0] if v.split() else ""
    # The footer (log_z etc.) comes after the data, so scan the whole file.
    with open(path) as f:
        for line in f:
            if line.startswith("#") and "=" in line:
                k, v = line[1:].strip().split("=", 1)
                meta[k] = v.split()[0] if v.split() else ""
    data = np.loadtxt(path, comments="#", ndmin=2)
    return names, data, meta


def main(path):
    names, data, meta = load(path)
    col = {n.lower(): i for i, n in enumerate(names)}
    params = data[:, [col[f"params--p{i}"] for i in range(toy.NDIM)]]

    if "log_weight" in col:
        w = np.exp(data[:, col["log_weight"]] - data[:, col["log_weight"]].max())
    elif "weight" in col:
        w = data[:, col["weight"]]
    else:
        w = np.ones(len(data))
    w = w / w.sum()

    ok = True

    # 1. Derived values on every row, and correct.
    exp_sum = params.sum(axis=1)
    exp_vec = params[:, [0]] * np.arange(1, toy.VEC_LEN + 1) + params[:, [1]]
    got_sum = data[:, col["derived--sum"]]
    got_vec = data[:, [col[f"derived--vec_{j}"] for j in range(toy.VEC_LEN)]]
    n_bad = int(np.sum(~np.isclose(got_sum, exp_sum, rtol=1e-6, atol=1e-8)))
    n_bad += int(
        np.sum(~np.all(np.isclose(got_vec, exp_vec, rtol=1e-6, atol=1e-8), axis=1))
    )
    print(f"rows: {len(data)}; rows with wrong/missing derived values: {n_bad}")
    ok &= n_bad == 0

    # 2. Moments (use the effective sample size for the tolerance).
    neff = 1.0 / np.sum(w**2)
    mean = w @ params
    std = np.sqrt(w @ (params - mean) ** 2)
    z = (mean - toy.MEAN) / (toy.SIGMA / np.sqrt(neff))
    print(f"n_eff ~ {neff:.0f}")
    print("mean  :", np.round(mean, 3), " truth:", toy.MEAN)
    print("std   :", np.round(std, 3), " truth:", toy.SIGMA)
    print("mean z-scores:", np.round(z, 2))
    ok &= bool(np.all(np.abs(z) < 5)) and bool(np.allclose(std, toy.SIGMA, rtol=0.1))

    # 3. Evidence.
    if "log_z" in meta:
        lz, lze = float(meta["log_z"]), float(meta.get("log_z_error", "nan"))
        print(f"log_z = {lz:.3f} +/- {lze:.3f}; analytic {toy.LOGZ:.3f}")
        ok &= abs(lz - toy.LOGZ) < max(0.1, 5 * lze)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
