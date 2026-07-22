#!/usr/bin/env python
"""
Post-hoc Weighted-Box-Fusion (WBF) ensemble for HafniaDataset prediction sets.

Fuses 2+ downloaded prediction JSONL files (AI City Track 6 / Hafnia "annotations.jsonl"
schema) at the box level, per class, per image, and writes a fused bundle
(annotations.jsonl + dataset_info.json) in the IDENTICAL schema (a core fused prediction file; platform-specific metadata may
need to be added separately).

NO model inference. Operates purely on already-downloaded prediction files.

Schema (verified on DG_annotations.jsonl / BASELINE_annotations.jsonl):
  Top-level record keys (order preserved on output):
    sample_index, height, width, split, tags, storage_format,
    collection_index, collection_id, remote_path, bboxes, dataset_name, file_path
  bbox keys (18, order preserved on output):
    height, width, top_left_x, top_left_y, area, class_name, class_idx,
    object_id, confidence, ground_truth, task_name, created_at, updated_at,
    meta, bboxes, classifications, polygons, bitmasks
  Boxes are NORMALIZED (top_left_x/y + width/height in [0,1]); class_idx in 0..9.
  Image alignment is by `remote_path` (sample_index 0..N-1 also aligns in practice).

WBF engine: uses `ensemble_boxes.weighted_boxes_fusion` if importable, else a
pure-numpy fallback that reproduces the same algorithm.

Usage:
  python wbf_ensemble.py \
      --inputs DG=/path/DG_annotations.jsonl BASELINE=/path/BASELINE_annotations.jsonl \
      --weights 1.0 1.0 \
      --dataset-info /path/dataset_info.json \
      --out /path/out_dir \
      --iou 0.55 --skip-box-thr 0.0 --conf-type box_and_model_avg \
      [--limit 200]    # only fuse first N images (subset test)

Output: <out_dir>/annotations.jsonl, <out_dir>/dataset_info.json,
        and <out_dir>.tar.gz packaging both (core prediction artifacts).
"""
import argparse
import json
import os
import sys
import tarfile
import time

import numpy as np

# ---- exact key order to reproduce on output ----
BBOX_KEY_ORDER = [
    "height", "width", "top_left_x", "top_left_y", "area", "class_name",
    "class_idx", "object_id", "confidence", "ground_truth", "task_name",
    "created_at", "updated_at", "meta", "bboxes", "classifications",
    "polygons", "bitmasks",
]
# default null-ish bbox template (everything except the geometry/class/conf fields)
BBOX_TEMPLATE = {
    "area": None,
    "object_id": None,
    "ground_truth": False,
    "task_name": "object_detection/predictions",
    "created_at": None,
    "updated_at": None,
    "meta": None,
    "bboxes": None,
    "classifications": None,
    "polygons": None,
    "bitmasks": None,
}

# ----------------------------------------------------------------------------
# WBF engine: prefer ensemble_boxes, else pure-numpy fallback
# ----------------------------------------------------------------------------
try:
    from ensemble_boxes import weighted_boxes_fusion as _eb_wbf
    _HAVE_EB = True
except Exception:
    _HAVE_EB = False


def _wbf_fallback(boxes_list, scores_list, labels_list, weights, iou_thr,
                  skip_box_thr, conf_type):
    """Pure-numpy WBF, single image. Mirrors ensemble_boxes algorithm.

    boxes_list[m]: list of [x1,y1,x2,y2] in [0,1] for model m.
    Returns (boxes Nx4, scores N, labels N).
    conf_type: 'avg' | 'max' | 'box_and_model_avg'
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    n_models = len(boxes_list)

    # gather all boxes into a flat list of (label, score*weight, score, box, model)
    entries = []  # per label
    by_label = {}
    for m in range(n_models):
        w = weights[m]
        for b, s, l in zip(boxes_list[m], scores_list[m], labels_list[m]):
            if s < skip_box_thr:
                continue
            by_label.setdefault(int(l), []).append(
                {"box": np.asarray(b, dtype=np.float64),
                 "score": float(s),
                 "w": float(w),
                 "model": m})

    out_boxes, out_scores, out_labels = [], [], []
    for label, items in by_label.items():
        # sort by weighted score desc
        items.sort(key=lambda e: e["score"] * e["w"], reverse=True)
        clusters = []  # each: dict with fused box + member list
        for it in items:
            best_iou, best_j = 0.0, -1
            for j, cl in enumerate(clusters):
                iou = _iou(cl["fused"], it["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou > iou_thr:
                clusters[best_j]["members"].append(it)
                _refit(clusters[best_j])
            else:
                clusters.append({"members": [it]})
                _refit(clusters[-1])

        for cl in clusters:
            mem = cl["members"]
            # weighted (by score*w) average box
            ws = np.array([e["score"] * e["w"] for e in mem])
            bx = np.stack([e["box"] for e in mem])
            fused_box = (bx * ws[:, None]).sum(0) / ws.sum()
            scores = np.array([e["score"] for e in mem])
            wlist = np.array([e["w"] for e in mem])
            n_present = len(set(e["model"] for e in mem))
            if conf_type == "max":
                conf = scores.max()
            elif conf_type == "avg":
                conf = (scores * wlist).sum() / wlist.sum()
            else:  # box_and_model_avg (ensemble_boxes default behaviour family)
                conf = (scores * wlist).sum() / wlist.sum()
                conf = conf * min(n_present, n_models) / float(sum(weights))
            out_boxes.append(fused_box)
            out_scores.append(float(conf))
            out_labels.append(label)

    if not out_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    return np.stack(out_boxes), np.array(out_scores), np.array(out_labels)


def _refit(cluster):
    mem = cluster["members"]
    ws = np.array([e["score"] * e["w"] for e in mem])
    bx = np.stack([e["box"] for e in mem])
    cluster["fused"] = (bx * ws[:, None]).sum(0) / ws.sum()


def _iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    iw = max(0.0, xb - xa); ih = max(0.0, yb - ya)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_wbf(boxes_list, scores_list, labels_list, weights, iou_thr,
            skip_box_thr, conf_type, engine):
    """Dispatch to ensemble_boxes or fallback. Returns (boxes, scores, labels)."""
    if engine == "fallback" or not _HAVE_EB:
        return _wbf_fallback(boxes_list, scores_list, labels_list, weights,
                             iou_thr, skip_box_thr, conf_type)
    # ensemble_boxes path
    if all(len(b) == 0 for b in boxes_list):
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    return _eb_wbf(boxes_list, scores_list, labels_list, weights=weights,
                   iou_thr=iou_thr, skip_box_thr=skip_box_thr,
                   conf_type=conf_type)


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def load_jsonl_index(path, limit=None):
    """Load records keyed by sample_index -> record (full), preserving order list.

    Alignment key is `sample_index` (verified unique: 0..N-1 in these sets).
    NOTE: `remote_path` is NOT safe as a key — the test set contains a duplicate
    remote_path on two distinct sample_index lines; keying by remote_path would
    collapse them and corrupt one image. sample_index is unique and stable.
    """
    by_key = {}
    order = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            key = r["sample_index"]
            if key in by_key:
                raise ValueError(
                    f"Duplicate sample_index {key} in {path}; cannot align safely")
            by_key[key] = r
            order.append(key)
    return by_key, order


def record_to_xyxy(record):
    """Extract per-class arrays from a record's bboxes (normalized xyxy)."""
    boxes, scores, labels, names = [], [], [], {}
    for b in record.get("bboxes") or []:
        x1 = b["top_left_x"]; y1 = b["top_left_y"]
        x2 = x1 + b["width"]; y2 = y1 + b["height"]
        # clip to [0,1] for WBF stability (input is already normalized)
        x1 = min(max(x1, 0.0), 1.0); y1 = min(max(y1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0); y2 = min(max(y2, 0.0), 1.0)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(b["confidence"]))
        labels.append(int(b["class_idx"]))
        names[int(b["class_idx"])] = b["class_name"]
    return boxes, scores, labels, names


def make_bbox(x1, y1, x2, y2, conf, class_idx, class_name):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    out = dict(BBOX_TEMPLATE)
    out["height"] = float(h)
    out["width"] = float(w)
    out["top_left_x"] = float(x1)
    out["top_left_y"] = float(y1)
    out["class_name"] = class_name
    out["class_idx"] = int(class_idx)
    out["confidence"] = float(conf)
    # reorder keys exactly
    return {k: out[k] for k in BBOX_KEY_ORDER}


def write_parquet(records, path):
    """Write annotations.parquet mirroring the original Hafnia bundle schema.

    Reproduces dtypes exactly:
      sample_index UInt64, height/width/collection_index Int64, tags List(String),
      bboxes List(Struct[18]) with Null-typed area/object_id/created_at/updated_at/
      meta/bboxes/classifications/polygons/bitmasks.
    """
    import polars as pl

    bbox_struct = pl.Struct([
        pl.Field("height", pl.Float64),
        pl.Field("width", pl.Float64),
        pl.Field("top_left_x", pl.Float64),
        pl.Field("top_left_y", pl.Float64),
        pl.Field("area", pl.Null),
        pl.Field("class_name", pl.String),
        pl.Field("class_idx", pl.Int64),
        pl.Field("object_id", pl.Null),
        pl.Field("confidence", pl.Float64),
        pl.Field("ground_truth", pl.Boolean),
        pl.Field("task_name", pl.String),
        pl.Field("created_at", pl.Null),
        pl.Field("updated_at", pl.Null),
        pl.Field("meta", pl.Null),
        pl.Field("bboxes", pl.Null),
        pl.Field("classifications", pl.Null),
        pl.Field("polygons", pl.Null),
        pl.Field("bitmasks", pl.Null),
    ])
    schema = {
        "sample_index": pl.UInt64,
        "height": pl.Int64,
        "width": pl.Int64,
        "split": pl.String,
        "tags": pl.List(pl.String),
        "storage_format": pl.String,
        "collection_index": pl.Int64,
        "collection_id": pl.String,
        "remote_path": pl.String,
        "bboxes": pl.List(bbox_struct),
        "dataset_name": pl.String,
        "file_path": pl.String,
    }
    df = pl.DataFrame(records, schema=schema, strict=False)
    df = df.select(list(schema.keys()))  # enforce column order
    df.write_parquet(path)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="NAME=path entries; the FIRST one is the primary "
                         "(its top-level record fields are copied to output).")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="per-model weights (same order/count as --inputs).")
    ap.add_argument("--dataset-info", required=True,
                    help="dataset_info.json to copy into the output bundle.")
    ap.add_argument("--carry", nargs="*", default=None,
                    help="extra files to copy verbatim into the bundle "
                         "(e.g. configuration.json environment.json). The full "
                         "Hafnia experiment-output bundle that the eval site "
                         "accepts contains 5 files; pass the metadata ones here.")
    ap.add_argument("--parquet", action="store_true",
                    help="also write annotations.parquet (mirror of jsonl) using "
                         "polars, matching the original Hafnia bundle layout.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--skip-box-thr", type=float, default=0.0,
                    help="drop input boxes below this conf before fusing")
    ap.add_argument("--conf-type", default="box_and_model_avg",
                    choices=["box_and_model_avg", "avg", "max"])
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "ensemble_boxes", "fallback"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N images (subset test)")
    ap.add_argument("--no-tar", action="store_true", help="skip tar.gz packaging")
    args = ap.parse_args()

    engine = args.engine
    if engine == "auto":
        engine = "ensemble_boxes" if _HAVE_EB else "fallback"
    elif engine == "ensemble_boxes" and not _HAVE_EB:
        print("[warn] ensemble_boxes not importable; using fallback")
        engine = "fallback"
    print(f"[info] WBF engine = {engine}  (ensemble_boxes available={_HAVE_EB})")

    # parse inputs
    named = []
    for spec in args.inputs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = os.path.basename(spec), spec
        named.append((name, path))
    weights = args.weights
    if weights is not None and len(weights) != len(named):
        sys.exit("--weights count must match --inputs count")
    if weights is None:
        weights = [1.0] * len(named)

    print(f"[info] inputs: {[(n,w) for (n,_),w in zip(named,weights)]}")
    print(f"[info] iou={args.iou} skip_box_thr={args.skip_box_thr} "
          f"conf_type={args.conf_type} limit={args.limit}")

    t0 = time.time()
    # load all
    indices = []
    orders = []
    for name, path in named:
        bp, order = load_jsonl_index(path, args.limit)
        indices.append(bp)
        orders.append(order)
        print(f"[info] loaded {name}: {len(bp)} images from {path}")

    primary_by_key = indices[0]   # keyed by sample_index
    primary_order = orders[0]

    # union of keys for class names (fallback name lookup)
    global_names = {}
    for bp in indices:
        for rec in bp.values():
            for b in rec.get("bboxes") or []:
                global_names[int(b["class_idx"])] = b["class_name"]
            break  # one record per set is enough to seed; extend below if missing
    # ensure complete: scan a bit more cheaply only if needed later

    os.makedirs(args.out, exist_ok=True)
    out_jsonl = os.path.join(args.out, "annotations.jsonl")

    n_img = 0
    in_box_counts = [0] * len(named)
    out_box_count = 0
    missing_in_secondary = 0
    out_records = [] if args.parquet else None

    with open(out_jsonl, "w") as fout:
        for key in primary_order:
            prec = primary_by_key[key]
            boxes_list, scores_list, labels_list = [], [], []
            for mi, bp in enumerate(indices):
                rec = bp.get(key)
                if rec is None:
                    if mi != 0:
                        missing_in_secondary += 1
                    boxes_list.append([]); scores_list.append([]); labels_list.append([])
                    continue
                b, s, l, nm = record_to_xyxy(rec)
                in_box_counts[mi] += len(b)
                for k, v in nm.items():
                    global_names.setdefault(k, v)
                boxes_list.append(b); scores_list.append(s); labels_list.append(l)

            fb, fs, fl = run_wbf(boxes_list, scores_list, labels_list,
                                 weights, args.iou, args.skip_box_thr,
                                 args.conf_type, engine)

            new_bboxes = []
            for box, sc, lab in zip(fb, fs, fl):
                ci = int(round(lab))
                cn = global_names.get(ci, f"class_{ci}")
                new_bboxes.append(make_bbox(box[0], box[1], box[2], box[3],
                                            sc, ci, cn))
            out_box_count += len(new_bboxes)

            # copy primary top-level record verbatim, replace bboxes
            outrec = dict(prec)
            outrec["bboxes"] = new_bboxes
            fout.write(json.dumps(outrec) + "\n")
            if out_records is not None:
                out_records.append(outrec)
            n_img += 1
            if n_img % 2000 == 0:
                print(f"  ... {n_img} images, {out_box_count} fused boxes, "
                      f"{time.time()-t0:.1f}s")

    # copy dataset_info.json
    with open(args.dataset_info) as f:
        di = json.load(f)
    out_di = os.path.join(args.out, "dataset_info.json")
    with open(out_di, "w") as f:
        json.dump(di, f, indent=4)

    bundle_files = [(out_jsonl, "annotations.jsonl"),
                    (out_di, "dataset_info.json")]

    # optional: annotations.parquet (mirror of jsonl) matching Hafnia bundle
    out_pq = None
    if args.parquet:
        out_pq = write_parquet(out_records, os.path.join(args.out,
                                                         "annotations.parquet"))
        bundle_files.append((out_pq, "annotations.parquet"))

    # optional: carry metadata files verbatim (configuration.json, environment.json)
    carried = []
    if args.carry:
        import shutil
        for src in args.carry:
            base = os.path.basename(src)
            dst = os.path.join(args.out, base)
            shutil.copyfile(src, dst)
            bundle_files.append((dst, base))
            carried.append(base)

    runtime = time.time() - t0
    print("\n========== WBF FUSION SUMMARY ==========")
    print(f"engine               : {engine}")
    print(f"images written       : {n_img}")
    for (name, _), c in zip(named, in_box_counts):
        print(f"input boxes [{name:>10}]: {c}")
    print(f"fused boxes (output) : {out_box_count}")
    if missing_in_secondary:
        print(f"[warn] images missing in a secondary set: {missing_in_secondary}")
    print(f"out annotations.jsonl: {out_jsonl}")
    print(f"out dataset_info.json: {out_di}")
    if out_pq:
        print(f"out annotations.parq : {out_pq}")
    if carried:
        print(f"carried metadata     : {carried}")
    print(f"runtime              : {runtime:.1f}s")

    if not args.no_tar:
        tarpath = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tarpath, "w:gz") as tf:
            for path, arc in bundle_files:
                tf.add(path, arcname=arc)
        print(f"bundle tar.gz        : {tarpath}  ({len(bundle_files)} files)")
    print("========================================")


if __name__ == "__main__":
    main()
