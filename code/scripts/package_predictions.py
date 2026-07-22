#!/usr/bin/env python
"""Package a prediction JSONL into a core prediction bundle (annotations only;
any platform-specific metadata files are added separately when required).

Applies the submission-packaging steps used for the reported results:
  - confidence floor (default 0.015)
  - per-class top-K truncation (default 100)
  - coordinate rounding (default 5 decimals)
  - confidence rounding (default 4 decimals)
and writes annotations.jsonl + annotations.parquet + dataset_info.json (+ .tar.gz).

These packaging steps are part of the evaluated submission pipeline; their
individual effect on the COCO AP was not measured separately.

Usage:
  python scripts/package_predictions.py IN.jsonl OUT_DIR \
      --confidence-floor 0.015 --per-class-top-k 100 --coordinate-decimals 5 \
      --confidence-decimals 4
"""
import argparse
import importlib.util
import json
import os
import shutil
import tarfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEOM = ("top_left_x", "top_left_y", "width", "height", "area")


def _load_wbf():
    # write_parquet lives in the sibling wbf_ensemble.py (script-relative import)
    spec = importlib.util.spec_from_file_location("wbf_ensemble", SCRIPT_DIR / "wbf_ensemble.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_jsonl", help="input predictions (Hafnia annotations.jsonl)")
    ap.add_argument("output_dir", help="output directory for the packaged bundle")
    ap.add_argument("--confidence-floor", type=float, default=0.015)
    ap.add_argument("--per-class-top-k", type=int, default=100)
    ap.add_argument("--coordinate-decimals", type=int, default=5)
    ap.add_argument("--confidence-decimals", type=int, default=4)
    a = ap.parse_args()

    floor, maxpc, ndec = a.confidence_floor, a.per_class_top_k, a.coordinate_decimals
    cdec = a.confidence_decimals
    os.makedirs(a.output_dir, exist_ok=True)
    records = []
    nin = nout = 0
    with open(a.input_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            bs = r.get("bboxes") or []
            nin += len(bs)
            bs = [b for b in bs if (b.get("confidence") or 0) >= floor]
            byc = {}
            for b in bs:
                byc.setdefault(b.get("class_idx"), []).append(b)
            kept = []
            for _c, lst in byc.items():
                lst.sort(key=lambda b: -(b.get("confidence") or 0))
                kept.extend(lst[:maxpc])
            for b in kept:
                for k in GEOM:
                    v = b.get(k)
                    if isinstance(v, float):
                        b[k] = round(v, ndec)
                c = b.get("confidence")
                if isinstance(c, float):
                    b["confidence"] = round(c, cdec)
            r["bboxes"] = kept
            nout += len(kept)
            records.append(r)

    with open(os.path.join(a.output_dir, "annotations.jsonl"), "w") as g:
        for r in records:
            g.write(json.dumps(r, separators=(",", ":")) + "\n")
    _load_wbf().write_parquet(records, os.path.join(a.output_dir, "annotations.parquet"))
    di = os.path.join(os.path.dirname(os.path.abspath(a.input_jsonl)), "dataset_info.json")
    if os.path.exists(di):
        shutil.copy(di, os.path.join(a.output_dir, "dataset_info.json"))
    tp = a.output_dir.rstrip("/") + ".tar.gz"
    with tarfile.open(tp, "w:gz") as t:
        for fn in ("dataset_info.json", "annotations.jsonl", "annotations.parquet"):
            fp = os.path.join(a.output_dir, fn)
            if os.path.exists(fp):
                t.add(fp, arcname=fn)
    print(f"in_boxes={nin} out_boxes={nout} ({nout / max(len(records), 1):.1f}/img) floor={floor}")
    print("tar:", tp)


if __name__ == "__main__":
    main()
