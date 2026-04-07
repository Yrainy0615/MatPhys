import argparse

import numpy as np
from plyfile import PlyData, PlyElement


def main():
    parser = argparse.ArgumentParser(
        description="Add zero f_rest_* SH fields so a simplified Gaussian PLY becomes 3DGS-compatible"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_sh_degree", type=int, default=3)
    args = parser.parse_args()

    ply = PlyData.read(args.input)
    v = ply["vertex"].data

    required = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    names = v.dtype.names
    missing = [n for n in required if n not in names]
    if missing:
        raise ValueError(f"Missing required fields in {args.input}: {missing}")

    n = len(v)
    extra_dim = 3 * ((args.max_sh_degree + 1) ** 2 - 1)

    dtype_full = [(name, "f4") for name in required]
    dtype_full += [(f"f_rest_{i}", "f4") for i in range(extra_dim)]

    out = np.empty(n, dtype=dtype_full)
    for name in required:
        out[name] = np.asarray(v[name], dtype=np.float32)
    for i in range(extra_dim):
        out[f"f_rest_{i}"] = 0.0

    PlyData([PlyElement.describe(out, "vertex")]).write(args.output)
    print(f"[done] wrote {args.output}")
    print(f"[info] points={n}, added_f_rest={extra_dim}")


if __name__ == "__main__":
    main()
