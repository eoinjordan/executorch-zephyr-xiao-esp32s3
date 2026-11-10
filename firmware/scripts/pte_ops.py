#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Print the operators of one or more ExecuTorch programs, comma separated.

The firmware embeds two programs. A kernel may be registered only once, so
the selective build needs the union of their operators per kernel library:

    pte_ops.py --exclude cortex_m:: a.pte b.pte

prints every operator of a.pte and b.pte that is not a cortex_m one, which is
the list for the portable kernels. The operators are read with the same tool
the ExecuTorch build uses for OPS_FROM_MODEL (codegen.tools.gen_oplist).
"""

import argparse
import contextlib
import io
import pathlib
import sys
import tempfile


def operators(pte: pathlib.Path) -> list[str]:
    import yaml
    from executorch.codegen.tools.gen_oplist import gen_oplist

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "ops.yaml"
        # gen_oplist prints progress to stdout; this script's stdout is the list.
        with contextlib.redirect_stdout(io.StringIO()):
            gen_oplist(output_path=str(out), model_file_path=str(pte))
        doc = yaml.safe_load(out.read_text())
    return list(doc.get("operators", {}))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pte", nargs="+", type=pathlib.Path)
    p.add_argument("--exclude", action="append", default=[], help="skip operators with this prefix")
    args = p.parse_args()

    names: list[str] = []
    for pte in args.pte:
        if not pte.exists():
            print(f"{pte}: not found", file=sys.stderr)
            return 1
        for op in operators(pte):
            if op not in names and not any(op.startswith(x) for x in args.exclude):
                names.append(op)
    print(",".join(sorted(names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
