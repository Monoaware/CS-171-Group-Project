#!/usr/bin/env python3
"""
CLIP-based dataset filter (local, offline after first weights download).

Decides whether to KEEP an image as "real car exterior" or MOVE it to a quarantine
folder with a reason: placeholder, ad_borders, cgi_like, interior, unreadable.

Features:
- Zero-shot prompts via open_clip (ViT-L/14 by default).
- Batch inference, GPU/CPU auto.
- Dry-run mode, mirrored subfolders, CSV logs for kept/rejected.
- Customizable thresholds.

Usage:
  python clip_filter.py \
      --root /data/cars_dataset \
      --rejects /data/quarantine/cars_run1 \
      --dry-run false \
      --batch-size 16
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import open_clip
from shutil import move

DEFAULT_MODEL = "ViT-L-14"
DEFAULT_PRETRAINED = "openai"
DEFAULT_LABELS = [
    "a photo of a real car exterior",
    "a photo of a car interior",
    "a 3D render of a car, CGI, digital model",
    "a generic placeholder icon of a car, not a real photo",
    "a photo of a car with advertisement banners around the border",
]
REJECT_MAP = {
    "a photo of a car interior": "interior",
    "a 3D render of a car, CGI, digital model": "cgi_like",
    "a generic placeholder icon of a car, not a real photo": "placeholder",
    "a photo of a car with advertisement banners around the border": "ad_borders",
}
DEFAULT_THRESHOLDS = {
    "keep_exterior_min": 0.27,
    "interior_min": 0.28,
    "cgi_like_min": 0.25,
    "placeholder_min": 0.23,
    "ad_borders_min": 0.23,
}

def find_images(root: Path, exts: Tuple[str,...]) -> List[Path]:
    imgs = []
    for ext in exts:
        imgs.extend(root.rglob(f"*{ext}"))
    return sorted([p for p in imgs if p.is_file()])

def ensure_quarantine_ok(root: Path, rej: Path):
    if root.resolve() in rej.resolve().parents:
        raise ValueError("Reject path must NOT be inside dataset root.")

def open_image_rgb(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError):
        return None

def move_to_reason(src_path: Path, root: Path, rej_root: Path, reason: str,
                   keep_subfolders: bool, dry_run: bool) -> Path:
    if keep_subfolders:
        dst_dir = rej_root / reason / src_path.parent.relative_to(root)
    else:
        dst_dir = rej_root / reason
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src_path.name
    if not dry_run:
        move(str(src_path), str(dst))
    return dst

class ClipZeroShot:
    def __init__(self, model_name: str, pretrained: str | Path, device: str):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_texts(self, labels: List[str]) -> torch.Tensor:
        text = self.tokenizer(labels).to(self.device)
        tfeat = self.model.encode_text(text)
        tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        if hasattr(self.model, "encode_image"):
            if batch.dtype != torch.float32:
                batch = batch.float()
            img_feat = self.model.encode_image(batch)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            return img_feat
        raise RuntimeError("Model has no encode_image method")

def decide(label_names: List[str], sims_row, thresholds: dict):
    idx = int(torch.argmax(sims_row).item())
    top_label = label_names[idx]
    top_score = float(sims_row[idx].item())

    if top_label == "a photo of a real car exterior" and top_score >= thresholds["keep_exterior_min"]:
        return top_label, top_score, "keep"

    for label, reason in REJECT_MAP.items():
        if top_label == label:
            key = reason + "_min"
            if key in thresholds and top_score >= thresholds[key]:
                return top_label, top_score, f"reject:{reason}"

    return top_label, top_score, "keep"

def main():
    ap = argparse.ArgumentParser(description="CLIP-based filter for car datasets (local).")
    ap.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    ap.add_argument("--rejects", type=Path, required=True, help="Quarantine folder for rejected images.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"CLIP model name (default: {DEFAULT_MODEL}).")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                    help="Pretrained tag or local checkpoint path (default: 'openai').")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if omitted).")
    ap.add_argument("--dry-run", type=lambda s: s.lower() in {'1','true','yes','y'}, default=False,
                    help="Print actions without moving files.")
    ap.add_argument("--keep-subfolders", type=lambda s: s.lower() in {'1','true','yes','y'}, default=True,
                    help="Mirror original subfolders inside reject root.")
    ap.add_argument("--exts", nargs="*", default=[".jpg",".jpeg",".png",".webp",".bmp"],
                    help="Image extensions to include.")
    ap.add_argument("--keep-exterior-min", type=float, default=DEFAULT_THRESHOLDS["keep_exterior_min"])
    ap.add_argument("--interior-min", type=float, default=DEFAULT_THRESHOLDS["interior_min"])
    ap.add_argument("--cgi-like-min", type=float, default=DEFAULT_THRESHOLDS["cgi_like_min"])
    ap.add_argument("--placeholder-min", type=float, default=DEFAULT_THRESHOLDS["placeholder_min"])
    ap.add_argument("--ad-borders-min", type=float, default=DEFAULT_THRESHOLDS["ad_borders_min"])
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        device = "mps" if mps_available else "cpu"
    
    root: Path = args.root
    rej_root: Path = args.rejects

    ensure_quarantine_ok(root, rej_root)
    rej_root.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "keep_exterior_min": args.keep_exterior_min,
        "interior_min": args.interior_min,
        "cgi_like_min": args.cgi_like_min,
        "placeholder_min": args.placeholder_min,
        "ad_borders_min": args.ad_borders_min,
    }

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.model}  |  Pretrained: {args.pretrained}")
    print(f"[INFO] Root: {root}")
    print(f"[INFO] Rejects: {rej_root}")
    print(f"[INFO] Dry run: {args.dry_run}")
    print(f"[INFO] Thresholds: {thresholds}")

    clip = ClipZeroShot(args.model, args.pretrained, device)
    with torch.no_grad():
        text_feats = clip.encode_texts(DEFAULT_LABELS)

    files = find_images(root, tuple(args.exts))
    if not files:
        print("[WARN] No images found.")
        return

    kept_csv = (rej_root / "kept.csv")
    rej_csv  = (rej_root / "rejections.csv")
    kept_fh = open(kept_csv, "w", newline="", encoding="utf-8")
    rej_fh  = open(rej_csv,  "w", newline="", encoding="utf-8")
    kept_log = csv.writer(kept_fh); kept_log.writerow(["path","top_label","score"])
    rej_log  = csv.writer(rej_fh);  rej_log.writerow(["reason","src_path","dst_path","top_label","score"])

    moved_counts = {v:0 for v in REJECT_MAP.values()}
    unreadable_count = 0
    kept_count = 0

    bs = max(1, args.batch_size)
    for i in tqdm(range(0, len(files), bs), desc="CLIP filtering"):
        batch_paths = files[i:i+bs]

        imgs: List[Image.Image] = []
        ok_idx = []
        for j, p in enumerate(batch_paths):
            img = open_image_rgb(p)
            if img is None:
                dst = move_to_reason(p, root, rej_root, "unreadable", args.keep_subfolders, args.dry_run)
                rej_log.writerow(["unreadable", str(p), str(dst), "unreadable", -1.0])
                unreadable_count += 1
            else:
                imgs.append(img); ok_idx.append(j)

        if not imgs:
            continue

        with torch.no_grad():
            img_feats = clip.encode_images(imgs)
            sims = (img_feats @ text_feats.T)

        for k, sim_row in enumerate(sims):
            p = batch_paths[ok_idx[k]]
            top_label, top_score, decision = decide(DEFAULT_LABELS, sim_row, thresholds)
            if decision == "keep":
                kept_log.writerow([str(p), top_label, f"{top_score:.6f}"])
                kept_count += 1
            else:
                reason = decision.split(":",1)[1]
                dst = move_to_reason(p, root, rej_root, reason, args.keep_subfolders, args.dry_run)
                rej_log.writerow([reason, str(p), str(dst), top_label, f"{top_score:.6f}"])
                moved_counts[reason] += 1

    kept_fh.close()
    rej_fh.close()

    print("\n=== Summary ===")
    print(f"Kept: {kept_count}")
    print(f"Unreadable moved: {unreadable_count}")
    for k,v in moved_counts.items():
        print(f"Rejected ({k}): {v}")
    print(f"\nLogs written to:\n  - {kept_csv}\n  - {rej_csv}")
    if args.dry_run:
        print("\n[NOTE] Dry-run was ON: no files were moved.")

if __name__ == "__main__":
    main()
