import os, json
from collections import defaultdict

ROOT = "/Users/khavinm/Desktop/Object detection/coco"  # <-- change if needed
ANN = os.path.join(ROOT, "annotations")
IMG = os.path.join(ROOT, "images")
LBL = os.path.join(ROOT, "labels")

SPLITS = [
    ("train2017", os.path.join(ANN, "instances_train2017.json")),
    ("val2017",   os.path.join(ANN, "instances_val2017.json")),
]

os.makedirs(LBL, exist_ok=True)

def load_json(p):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing annotations: {p}")
    with open(p, "r") as f:
        return json.load(f)

# Build a global category id mapping (old COCO IDs -> 0..N-1)
cats = {}
for _, annp in SPLITS:
    if os.path.isfile(annp):
        d = load_json(annp)
        for c in d.get("categories", []):
            cats[c["id"]] = c["name"]
old_ids = sorted(cats.keys())
idmap = {old: i for i, old in enumerate(old_ids)}
names = [cats[i] for i in old_ids]
print(f"Classes: {len(names)} -> {names}")

def convert_split(split, annp):
    data = load_json(annp)
    images = {im["id"]: im for im in data.get("images", [])}
    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        if a.get("iscrowd", 0) == 1:
            continue
        if "bbox" not in a or a.get("category_id") not in idmap:
            continue
        anns_by_img[a["image_id"]].append(a)

    out_dir = os.path.join(LBL, split)
    os.makedirs(out_dir, exist_ok=True)

    n_files, n_boxes = 0, 0
    for img_id, im in images.items():
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            continue
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        out_path = os.path.join(out_dir, stem + ".txt")
        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, bw, bh = a["bbox"]  # COCO x,y,w,h
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            ww = bw / w
            hh = bh / h
            cls = idmap[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        n_files += 1
        n_boxes += len(lines)
    print(f"[{split}] wrote {n_files} label files, {n_boxes} boxes -> {out_dir}")

# Convert the splits that exist
for split, annp in SPLITS:
    if os.path.isfile(annp):
        convert_split(split, annp)
    else:
        print(f"[{split}] skipped (no {os.path.basename(annp)})")

# Minimal YAML (points to images/*2017 and labels/*2017)
yaml_path = os.path.join(ROOT, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {ROOT}\n")
    f.write("train: images/train2017\n")
    f.write("val: images/val2017\n")
    f.write("names:\n")
    for i, n in enumerate(names):
        n = str(n).replace(":", "-")
        f.write(f"  {i}: {n}\n")
print(f"[data.yaml] -> {yaml_path}")
