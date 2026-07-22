"""In-process Weighted Box Fusion (WBF) for ensembling HafniaDataset prediction
files WITHIN a single benchmark inference pipeline.

AI City Track 6 rule: "Ensembles are allowed, provided that they can be executed
as a single inference pipeline within the platform constraints." So the submitted
ensemble is produced by ONE benchmark.py run that infers each member + view and
fuses them here (no separate experiments / no off-platform fusion of the SUBMITTED
file). The standalone CLI (scripts/wbf_ensemble.py) shares this exact algorithm and
is used for offline parameter tuning on downloaded predictions. Neither is
part of the final configuration set reported in the paper.

Algorithm/schema verified on DG_annotations.jsonl / BASELINE_annotations.jsonl
(14,925 images): normalized xyxy, class_idx 0..9, alignment by sample_index.
"""
import json
import os

import numpy as np

# exact bbox key order to reproduce on output (Hafnia annotations.jsonl schema)
BBOX_KEY_ORDER = [
    "height", "width", "top_left_x", "top_left_y", "area", "class_name",
    "class_idx", "object_id", "confidence", "ground_truth", "task_name",
    "created_at", "updated_at", "meta", "bboxes", "classifications",
    "polygons", "bitmasks",
]
BBOX_TEMPLATE = {
    "area": None, "object_id": None, "ground_truth": False,
    "task_name": "object_detection/predictions", "created_at": None,
    "updated_at": None, "meta": None, "bboxes": None, "classifications": None,
    "polygons": None, "bitmasks": None,
}

try:
    from ensemble_boxes import weighted_boxes_fusion as _eb_wbf
    _HAVE_EB = True
except Exception:
    _HAVE_EB = False


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


def _refit(cluster):
    mem = cluster["members"]
    ws = np.array([e["score"] * e["w"] for e in mem])
    bx = np.stack([e["box"] for e in mem])
    cluster["fused"] = (bx * ws[:, None]).sum(0) / ws.sum()


def _wbf_fallback(boxes_list, scores_list, labels_list, weights, iou_thr,
                  skip_box_thr, conf_type):
    """Pure-numpy WBF, single image. Mirrors the ensemble_boxes algorithm."""
    if weights is None:
        weights = [1.0] * len(boxes_list)
    n_models = len(boxes_list)
    by_label = {}
    for m in range(n_models):
        w = weights[m]
        for b, s, l in zip(boxes_list[m], scores_list[m], labels_list[m]):
            if s < skip_box_thr:
                continue
            by_label.setdefault(int(l), []).append(
                {"box": np.asarray(b, dtype=np.float64), "score": float(s),
                 "w": float(w), "model": m})
    out_boxes, out_scores, out_labels = [], [], []
    for label, items in by_label.items():
        items.sort(key=lambda e: e["score"] * e["w"], reverse=True)
        clusters = []
        for it in items:
            best_iou, best_j = 0.0, -1
            for j, cl in enumerate(clusters):
                iou = _iou(cl["fused"], it["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou > iou_thr:
                clusters[best_j]["members"].append(it); _refit(clusters[best_j])
            else:
                clusters.append({"members": [it]}); _refit(clusters[-1])
        for cl in clusters:
            mem = cl["members"]
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
            else:  # box_and_model_avg
                conf = (scores * wlist).sum() / wlist.sum()
                conf = conf * min(n_present, n_models) / float(sum(weights))
            out_boxes.append(fused_box); out_scores.append(float(conf)); out_labels.append(label)
    if not out_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    return np.stack(out_boxes), np.array(out_scores), np.array(out_labels)


def run_wbf(boxes_list, scores_list, labels_list, weights, iou_thr,
            skip_box_thr, conf_type, engine):
    if engine == "fallback" or not _HAVE_EB:
        return _wbf_fallback(boxes_list, scores_list, labels_list, weights,
                             iou_thr, skip_box_thr, conf_type)
    if all(len(b) == 0 for b in boxes_list):
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    return _eb_wbf(boxes_list, scores_list, labels_list, weights=weights,
                   iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)


def load_jsonl_index(path):
    """Load records keyed by unique sample_index (safe alignment key)."""
    by_key, order = {}, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = r["sample_index"]
            if key in by_key:
                raise ValueError(f"Duplicate sample_index {key} in {path}")
            by_key[key] = r
            order.append(key)
    return by_key, order


def record_to_xyxy(record):
    boxes, scores, labels, names = [], [], [], {}
    for b in record.get("bboxes") or []:
        x1 = b["top_left_x"]; y1 = b["top_left_y"]
        x2 = x1 + b["width"]; y2 = y1 + b["height"]
        x1 = min(max(x1, 0.0), 1.0); y1 = min(max(y1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0); y2 = min(max(y2, 0.0), 1.0)
        boxes.append([x1, y1, x2, y2]); scores.append(float(b["confidence"]))
        labels.append(int(b["class_idx"])); names[int(b["class_idx"])] = b["class_name"]
    return boxes, scores, labels, names


def make_bbox(x1, y1, x2, y2, conf, class_idx, class_name, task_name="object_detection/predictions"):
    out = dict(BBOX_TEMPLATE)
    out["height"] = float(max(0.0, y2 - y1)); out["width"] = float(max(0.0, x2 - x1))
    out["top_left_x"] = float(x1); out["top_left_y"] = float(y1)
    out["class_name"] = class_name; out["class_idx"] = int(class_idx)
    out["confidence"] = float(conf); out["task_name"] = task_name
    return {k: out[k] for k in BBOX_KEY_ORDER}


def fuse_annotation_files(input_jsonls, out_jsonl, weights=None, iou_thr=0.55,
                          skip_box_thr=0.0, conf_type="box_and_model_avg", engine="auto"):
    """Fuse N annotations.jsonl prediction files (same schema) per image via WBF.

    The FIRST file is primary: its top-level per-record fields are copied to the
    output, with only `bboxes` replaced by the fused boxes. Returns a stats dict.
    """
    eng = ("ensemble_boxes" if _HAVE_EB else "fallback") if engine == "auto" else engine
    if eng == "ensemble_boxes" and not _HAVE_EB:
        eng = "fallback"
    if weights is None:
        weights = [1.0] * len(input_jsonls)

    indices, orders = [], []
    for p in input_jsonls:
        bp, order = load_jsonl_index(p)
        indices.append(bp); orders.append(order)
    primary_by_key, primary_order = indices[0], orders[0]

    # Carry the SAME prediction task name the single-model path emits (don't hardcode):
    # read it from the primary file's first prediction box.
    pred_task_name = "object_detection/predictions"
    for rec in primary_by_key.values():
        bbs = rec.get("bboxes") or []
        if bbs and bbs[0].get("task_name"):
            pred_task_name = bbs[0]["task_name"]
            break
    # Count primary images missing from any secondary member (incomplete/corrupt outputs).
    primary_keys = set(primary_order)
    missing_secondary = sum(len(primary_keys - set(bp.keys())) for bp in indices[1:])

    global_names = {}
    for bp in indices:
        for rec in bp.values():
            for b in rec.get("bboxes") or []:
                global_names[int(b["class_idx"])] = b["class_name"]
            break

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    n_img = 0; in_counts = [0] * len(input_jsonls); out_count = 0
    with open(out_jsonl, "w") as fout:
        for key in primary_order:
            prec = primary_by_key[key]
            bl, sl, ll = [], [], []
            for mi, bp in enumerate(indices):
                rec = bp.get(key)
                if rec is None:
                    bl.append([]); sl.append([]); ll.append([]); continue
                b, s, l, nm = record_to_xyxy(rec)
                in_counts[mi] += len(b)
                for k, v in nm.items():
                    global_names.setdefault(k, v)
                bl.append(b); sl.append(s); ll.append(l)
            fb, fs, fl = run_wbf(bl, sl, ll, weights, iou_thr, skip_box_thr, conf_type, eng)
            nb = [make_bbox(bx[0], bx[1], bx[2], bx[3], sc, int(round(lb)),
                            global_names.get(int(round(lb)), f"class_{int(round(lb))}"),
                            task_name=pred_task_name)
                  for bx, sc, lb in zip(fb, fs, fl)]
            out_count += len(nb)
            outrec = dict(prec); outrec["bboxes"] = nb
            fout.write(json.dumps(outrec) + "\n"); n_img += 1
    return {"engine": eng, "images": n_img, "in_boxes": in_counts, "out_boxes": out_count,
            "missing_secondary_keys": missing_secondary, "task_name": pred_task_name}
