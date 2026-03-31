#!/usr/bin/env python3
"""Execute notebook cells sequentially, capturing output."""
import json, sys, os, traceback, io, contextlib

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks"))

nb_path = "10_rq2_esrs_context_ablation.ipynb"
nb = json.load(open(nb_path))

# Shared namespace
ns = {"__name__": "__main__"}

code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
total = len(code_cells)

for idx, (cell_num, cell) in enumerate(code_cells):
    src = "".join(cell["source"])
    print(f"\n{'='*60}")
    print(f"CELL {cell_num} ({idx+1}/{total})")
    print(f"{'='*60}")
    # Show first 2 lines
    lines = src.split("\n")
    for l in lines[:2]:
        print(f"  > {l[:100]}")
    if len(lines) > 2:
        print(f"  > ... ({len(lines)} lines)")
    print()

    # Capture stdout
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"<cell_{cell_num}>", "exec"), ns)
        output = buf.getvalue()
        if output:
            # Limit output
            lines_out = output.split("\n")
            if len(lines_out) > 50:
                print("\n".join(lines_out[:30]))
                print(f"... ({len(lines_out) - 50} lines omitted) ...")
                print("\n".join(lines_out[-20:]))
            else:
                print(output)
    except Exception as e:
        output = buf.getvalue()
        if output:
            print(output)
        print(f"\n*** ERROR in cell {cell_num}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

print(f"\n{'='*60}")
print("ALL CELLS EXECUTED SUCCESSFULLY")
print(f"{'='*60}")
