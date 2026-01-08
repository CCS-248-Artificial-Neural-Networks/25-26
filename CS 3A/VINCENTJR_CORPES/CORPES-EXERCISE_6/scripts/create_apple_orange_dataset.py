#!/usr/bin/env python3
"""
Create an Apple vs Orange dataset from a raw Fruit Images Dataset clone.

Usage:
  python scripts/create_apple_orange_dataset.py --src archive/raw_fruits --out archive/apple_orange_split --train 0.7 --val 0.15 --test 0.15 --seed 42

The script looks for class folders containing the keywords 'Apple' and 'Orange'
in the `Training` and `Test` folders of the source repo, copies images into
two class folders (`apple`, `orange`) and splits them into train/val/test.
"""
import argparse
from pathlib import Path
import random
import shutil


def parse_args():
    p = argparse.ArgumentParser(description='Create apple vs orange dataset and split it')
    p.add_argument('--src', required=True, help='Path to raw fruits repo clone')
    p.add_argument('--out', required=True, help='Output base folder for split dataset')
    p.add_argument('--train', type=float, default=0.7, help='Train fraction')
    p.add_argument('--val', type=float, default=0.15, help='Validation fraction')
    p.add_argument('--test', type=float, default=0.15, help='Test fraction')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def gather_images(src_root: Path, keywords):
    imgs = []
    # Look into Training and Test folders if present
    for sub in ('Training', 'Test'):
        dirp = src_root / sub
        if not dirp.exists():
            continue
        for class_dir in dirp.iterdir():
            if not class_dir.is_dir():
                continue
            name = class_dir.name.lower()
            if any(k in name for k in keywords):
                for p in class_dir.rglob('*'):
                    if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        imgs.append(p)
    return imgs


def split_and_copy(imgs, dest_base: Path, subset_name, seed):
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    for i, p in enumerate(imgs):
        dest_dir = dest_base / subset_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest_dir / p.name)
    return n


def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print('Source folder does not exist:', src)
        return

    # Keywords to match apple and orange classes
    apple_kw = ['apple']
    orange_kw = ['orange']

    print('Gathering apple images...')
    apple_imgs = gather_images(src, apple_kw)
    print(f'Found {len(apple_imgs)} apple images')

    print('Gathering orange images...')
    orange_imgs = gather_images(src, orange_kw)
    print(f'Found {len(orange_imgs)} orange images')

    if len(apple_imgs) == 0 or len(orange_imgs) == 0:
        print('Not enough images found for one of the classes. Check the source dataset structure.')
        return

    # Prepare per-class folders (raw) and then split
    raw_base = Path(str(out) + '_raw')
    # clear raw_base
    if raw_base.exists():
        shutil.rmtree(raw_base)
    (raw_base / 'apple').mkdir(parents=True, exist_ok=True)
    (raw_base / 'orange').mkdir(parents=True, exist_ok=True)

    for p in apple_imgs:
        shutil.copy2(p, raw_base / 'apple' / p.name)
    for p in orange_imgs:
        shutil.copy2(p, raw_base / 'orange' / p.name)

    # Now split each class
    if Path(out).exists():
        shutil.rmtree(out)
    for cls in ('apple', 'orange'):
        imgs = sorted((raw_base / cls).iterdir())
        random.Random(args.seed).shuffle(imgs)
        n = len(imgs)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        train = imgs[:n_train]
        val = imgs[n_train:n_train + n_val]
        test = imgs[n_train + n_val:]
        for subset_name, files in [('train', train), ('val', val), ('test', test)]:
            dest_dir = Path(out) / subset_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)
        print(f'Class {cls}: train={len(train)}, val={len(val)}, test={len(test)}')
    print('Done. Output at', out)


if __name__ == '__main__':
    main()
