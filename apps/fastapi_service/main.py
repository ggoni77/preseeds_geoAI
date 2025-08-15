# apps/fastapi_service/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi import BackgroundTasks
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
import os, io, uuid, zipfile, time
from typing import List, Tuple, Dict

import numpy as np
import cv2
from PIL import Image

# =========================
# Configuración básica
# =========================
app = FastAPI(title="PreSeeds · Uniformidad API", version="0.2.0")
OUT_DIR = os.getenv("OUT_DIR", "/app/outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Utilidades de imagen
# =========================
def _read_rgb(image_bytes: bytes, max_side: int = 4000) -> np.ndarray:
    """
    Lee bytes a imagen RGB (uint8). Si es gigantesca, reduce manteniendo aspecto
    para evitar OOM/timeouts en Render Free.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        img.thumbnail((max_side, max_side))  # mantiene aspecto
    return np.array(img)  # RGB uint8


def _clahe_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normaliza iluminación aplicando CLAHE sobre el canal L en LAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _compute_indices(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula ExG y VARI y retorna normalizados a [0,1]."""
    rgb_f = rgb.astype(np.float32) / 255.0
    R, G, B = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    exg = 2 * G - R - B
    vari = (G - R) / (G + R - B + 1e-6)
    exg_n = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    vari_n = (vari - vari.min()) / (vari.max() - vari.min() + 1e-6)
    return exg_n.astype(np.float32), vari_n.astype(np.float32)


def _green_rules(rgb: np.ndarray) -> np.ndarray:
    """
    Máscara para píxeles verdes usando HSV y Lab.
    - HSV: H ~35..95°, S >= 0.15, V >= 0.12
    - Lab: a* negativo → verdoso
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0  # 0..360 aprox
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    mask_hsv = (h >= 35) & (h <= 95) & (s >= 0.15) & (v >= 0.12)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    a = lab[..., 1].astype(np.int16) - 128
    mask_lab = (a < 0)

    return (mask_hsv & mask_lab)


def _local_percentile_threshold(score: np.ndarray, tile: int = 128, q: int = 65) -> np.ndarray:
    """
    Umbral local por baldosa: cada tile usa su propio percentil como threshold.
    Evita que sombras o bandas arruinen un umbral global.
    """
    h, w = score.shape
    mask = np.zeros_like(score, dtype=bool)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            ys, xs = slice(y, min(y + tile, h)), slice(x, min(x + tile, w))
            block = score[ys, xs]
            t = np.percentile(block, q)
            mask[ys, xs] = block >= t
    return mask


def _postprocess(mask: np.ndarray, min_area: int = 300) -> np.ndarray:
    """
    Limpieza morfológica:
    - abrir/cerrar
    - quitar componentes pequeñas
    - rellenar huecos grandes
    """
    m = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
    out = np.isin(labels, np.where(keep)[0]).astype(np.uint8) * 255

    inv = 255 - out
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k, iterations=1)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k, iterations=1)
    out = 255 - inv

    return (out > 127)


def segment_cultivo(rgb: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Pipeline principal: CLAHE → ExG/VARI → threshold local → reglas verdes → postproceso.
    Devuelve mask booleana (cultivo) y métrica de cobertura.
    """
    rgb2 = _clahe_rgb(rgb)
    exg, vari = _compute_indices(rgb2)
    score = 0.6 * exg + 0.4 * vari
    mask_idx = _local_percentile_threshold(score, tile=128, q=65)
    mask_green = _green_rules(rgb2)
    mask = mask_idx & mask_green
    mask = _postprocess(mask, min_area=300)
    coverage = float(mask.mean())  # 0..1
    return mask, {"coverage": coverage}


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Pinta cultivo en verde translucido sobre la imagen original."""
    overlay = rgb.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[mask] = (alpha * color + (1 - alpha) * overlay[mask]).astype(np.uint8)
    return overlay


def infer_image_bytes(image_bytes: bytes, out_dir: str) -> Dict:
    """
    Corre el pipeline sobre una imagen (bytes) y guarda:
      - mask.png
      - overlay.jpg
    Retorna dict con run_id, paths y métricas.
    """
    os.makedirs(out_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]

    rgb = _read_rgb(image_bytes, max_side=4000)
    mask, metrics = segment_cultivo(rgb)
    overlay = make_overlay(rgb, mask)

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_path = os.path.join(out_dir, f"{run_id}_mask.png")
    over_path = os.path.join(out_dir, f"{run_id}_overlay.jpg")

    cv2.imwrite(mask_path, mask_u8)
    cv2.imwrite(over_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    return {
        "run_id": run_id,
        "mask_path": mask_path,
        "overlay_path": over_path,
        "metrics": metrics,
    }


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    # Redirige al swagger para evitar 404
    return RedirectResponse(url="/docs")


@app.post("/uniformidad/infer")
async def uniformidad_infer(
    images: List[UploadFile] = File(..., description="1..N imágenes (JPG/PNG)"),
    assessor: str = Form(""),
    campo: str = Form(""),
    cultivo: str = Form(""),
    lote: str = Form(""),
    localidad: str = Form(""),
    area_img_m2: str = Form(""),
    blocks: int = Form(10),
):
    """
    Procesa N imágenes, guarda resultados y devuelve un ZIP con:
    - img_XX_mask.png
    - img_XX_overlay.jpg
    """
    if not images:
        raise HTTPException(status_code=400, detail="Sube al menos una imagen")

    # Procesamos todas y armamos el ZIP
    run_id = None
    tmp_files = []
    zip_path = os.path.join(OUT_DIR, f"{uuid.uuid4().hex[:8]}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(images, start=1):
            content = await f.read()
            result = infer_image_bytes(content, OUT_DIR)
            if run_id is None:
                run_id = result["run_id"]
                # renombrar el zip con run_id definitivo
                zf.close()
                new_zip = os.path.join(OUT_DIR, f"{run_id}.zip")
                os.rename(zip_path, new_zip)
                zip_path = new_zip
                zf = zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED)

            # agregar archivos al ZIP
            arc_mask = f"img_{i:02d}_mask.png"
            arc_over = f"img_{i:02d}_overlay.jpg"
            zf.write(result["mask_path"], arcname=arc_mask)
            zf.write(result["overlay_path"], arcname=arc_over)
            tmp_files += [result["mask_path"], result["overlay_path"]]

        # (Opcional) incluir un pequeño README con metadatos
        meta_txt = (
            "PreSeeds · Uniformidad\n"
            f"assessor={assessor}\n"
            f"campo={campo}\n"
            f"cultivo={cultivo}\n"
            f"lote={lote}\n"
            f"localidad={localidad}\n"
            f"area_img_m2={area_img_m2}\n"
            f"blocks={blocks}\n"
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        zf.writestr("metadata.txt", meta_txt)

    # (Opcional) borrar temporales para ahorrar espacio
    # for p in tmp_files:
    #     try: os.remove(p)
    #     except: pass

    return {"run_id": run_id, "zip_path": zip_path, "message": "OK"}


@app.get("/download/{run_id}")
def download(run_id: str):
    zip_path = os.path.join(OUT_DIR, f"{run_id}.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="No encontrado o aún procesando")
    return FileResponse(zip_path, media_type="application/zip", filename=f"{run_id}.zip")
