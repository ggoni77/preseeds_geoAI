import cv2
import numpy as np
from typing import Tuple

def _gray_world(img_bgr: np.ndarray) -> np.ndarray:
    # Balance gris-mundo para estabilizar colores bajo distinta luz
    img = img_bgr.astype(np.float32)
    mean = img.reshape(-1,3).mean(axis=0) + 1e-6
    scale = mean.mean() / mean
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def _clahe(img_bgr: np.ndarray) -> np.ndarray:
    # Aumenta contraste en espacios con sombra
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def _th_otsu(x: np.ndarray) -> np.ndarray:
    x8 = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(x8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def veg_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segmentación robusta cultivo vs. suelo:
    1) corrección de iluminación (gray-world + CLAHE)
    2) 4 índices (VARI, ExG, ExGR, CIVE)
    3) umbral Otsu por índice
    4) voto mayoritario (>=3 de 4)
    5) morfología y limpieza por área
    """
    img = _clahe(_gray_world(img_bgr))
    b,g,r = cv2.split(img.astype(np.float32))

    # Índices
    vari = (g - r) / (g + r - b + 1e-6)
    exg  = 2*g - r - b
    exgr = exg - (1.4*r - g)
    cive = 0.441*r - 0.811*g + 0.385*b + 18.787  # menor es +verde; invertimos luego

    # Umbrales (normalizamos + Otsu)
    th_vari = _th_otsu(vari)
    th_exg  = _th_otsu(exg)
    th_exgr = _th_otsu(exgr)
    th_cive = _th_otsu(-cive)   # invertido

    votes = (th_vari>0).astype(np.uint8) + (th_exg>0).astype(np.uint8) + \
            (th_exgr>0).astype(np.uint8) + (th_cive>0).astype(np.uint8)

    mask = (votes >= 3).astype(np.uint8)*255

    # Morfología + limpieza por área mínima
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # remover manchas pequeñas < 0.02% del frame
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        min_area = max(10, int(0.0002 * mask.size))
        keep = np.zeros(num_labels-1, dtype=bool)
        keep[areas >= min_area] = True
        clean = np.zeros_like(mask)
        for idx,k in enumerate(keep, start=1):
            if k: clean[labels==idx] = 255
        mask = clean
    return mask

def split_top_bottom(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = img.shape[0]
    return img[:h//2, :], img[h//2:, :]

def detect_rows(bottom_bgr: np.ndarray, mask: np.ndarray):
    masked = cv2.bitwise_and(bottom_bgr, bottom_bgr, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)
    return lines if lines is not None else []

def block_stats(mask: np.ndarray, blocks: int = 10):
    h, w = mask.shape
    bh, bw = h // blocks, w // blocks
    stats = []
    for i in range(blocks):
        for j in range(blocks):
            y0, y1 = i*bh, (i+1)*bh if i < blocks-1 else h
            x0, x1 = j*bw, (j+1)*bw if j < blocks-1 else w
            tile = mask[y0:y1, x0:x1]
            area = tile.size
            cultivo = int(tile.sum() // 255)
            pct = 100.0 * cultivo / area
            label = 'Alta' if pct > 60 else ('Media' if pct >= 30 else 'Baja')
            stats.append({'row': i,'col': j,'pct_cultivo': round(pct,2),'clase': label})
    return stats

def render_processed(img_bgr: np.ndarray, mask_bottom: np.ndarray) -> np.ndarray:
    top, bottom = split_top_bottom(img_bgr)
    bottom_rgb = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
    overlay = np.zeros_like(bottom_rgb)
    overlay[mask_bottom==255] = (255,255,0)   # cultivo
    overlay[mask_bottom==0]   = (255,0,255)   # suelo
    blended = cv2.addWeighted(bottom_rgb, 1.0, overlay, 0.35, 0)
    out = np.vstack([cv2.cvtColor(top, cv2.COLOR_BGR2RGB), blended])
    return out
