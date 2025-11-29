#load all images from data/data_contrast and return as numpy array
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
DATA_CONTRAST_DIR = Path(__file__).resolve().parent.parent / 'data' / 'data_contrast'
DATA_INPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'data1'
IMAGE_SIZE = (600, 600)

def load_images_to_array(input_dir, exts=IMAGE_EXTS):
    input_dir = Path(input_dir)
    images = []  # lista (path, PIL.Image)
    for p in input_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                img = Image.open(p).convert('RGBA') if p.suffix.lower() == '.png' else Image.open(p).convert('RGB')
                images.append((p, img))
                print(f"IMG {p}, rozmiar: {img.size}, tryb: {img.mode}")
            except Exception as e:
                print(f"Warning: nie można wczytać {p}: {e}")
    return images

def change_contrast_pil(image_pil, factor):
    # image_pil: PIL.Image
    enhancer = ImageEnhance.Contrast(image_pil)
    return enhancer.enhance(factor)

def process_and_save(contrast_factor=1.5, exts=IMAGE_EXTS, mirror_structure=True, overwrite=False):
    input_dir = DATA_INPUT_DIR
    output_dir = DATA_CONTRAST_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Przetwarzanie obrazów z {input_dir} i zapisywanie do {output_dir} z kontrastem {contrast_factor}")

    for p in input_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            img = Image.open(p).convert('RGBA') if p.suffix.lower() == '.png' else Image.open(p).convert('RGB')
            print(f"IMG {p}, rozmiar: {img.size}, tryb: {img.mode}")
        
        rel = p.relative_to(input_dir) if mirror_structure else p.name
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            print(f"Pominięto (już istnieje): {out_path}")
            continue

        try:
            img_mod = change_contrast_pil(img, contrast_factor)

            save_kwargs = {}
            fmt = p.suffix.lower().lstrip('.').upper()
            if fmt == 'JPG':
                fmt = 'JPEG'
                save_kwargs['quality'] = 95
            if fmt == 'PNG':
                save_kwargs['compress_level'] = 6

            img_mod.save(out_path, format=fmt, **save_kwargs)
            print(f"SAVED IMG: {out_path}")

        except Exception as e:
            print(f"Błąd przy przetwarzaniu {p}: {e}")

input_dir = 'data/data1'      
contrast_factor = 1.6          

process_and_save(contrast_factor)