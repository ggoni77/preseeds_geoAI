from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
import json
import traceback
import io
import time

# ----------------------------
# Config básica
# ----------------------------
app = FastAPI(
    title="PreSeeds • Uniformidad API",
    description="API para procesar imágenes de uniformidad de siembra (subida web + jobs en background).",
    version="0.3.0",
)

# CORS amplio para poder usar desde cualquier UI simple
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Folders persistentes dentro del container (Render)
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # .../apps/..
OUT_DIR = BASE_DIR / "outputs"
TMP_DIR = BASE_DIR / "tmp"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

JOBS_DB = OUT_DIR / "jobs.json"  # persistimos un índice mínimo de estados

def _load_jobs() -> dict:
    if JOBS_DB.exists():
        try:
            return json.loads(JOBS_DB.read_text())
        except Exception:
            return {}
    return {}

def _save_jobs(data: dict) -> None:
    JOBS_DB.write_text(json.dumps(data, indent=2))

# Estructura mínima de job
# {
#   <run_id>: {
#       "status": "pending|running|done|error",
#       "created_at": epoch,
#       "meta": { ... },
#       "zip_path": "/app/outputs/<run_id>.zip",
#       "error": "... (si error)"
#   }
# }

# ----------------------------
# Utilidad de "procesamiento"
# (MVP: crea una versión procesada + ZIP)
# ----------------------------
def process_images(run_id: str, img_paths: List[Path], blocks: int, meta: dict):
    jobs = _load_jobs()
    jobs[run_id]["status"] = "running"
    _save_jobs(jobs)

    work_dir = TMP_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        processed_paths = []

        # Ejemplo de "proceso": abrir, convertir a RGB, agregar una franja semitransparente y guardar
        for idx, src in enumerate(img_paths, start=1):
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size

                # Overlay simple (simula segmentación/RESULTADO MVP)
                overlay = Image.new("RGBA", (w, h), (0, 255, 0, 60))
                im_rgba = im.copy().convert("RGBA")
                im_proc = Image.alpha_composite(im_rgba, overlay).convert("RGB")

                out_path = work_dir / f"processed_{idx:03d}.jpg"
                im_proc.save(out_path, quality=90)
                processed_paths.append(out_path)

        # Guardamos un pequeño JSON de resultado + metadatos
        (work_dir / "result.json").write_text(json.dumps({
            "run_id": run_id,
            "blocks": blocks,
            "meta": meta,
            "images_count": len(processed_paths),
        }, indent=2))

        # Empaquetamos todo en un ZIP final
        zip_path = OUT_DIR / f"{run_id}.zip"
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
            for p in processed_paths:
                zf.write(p, arcname=p.name)
            zf.write(work_dir / "result.json", arcname="result.json")

        jobs = _load_jobs()
        jobs[run_id]["status"] = "done"
        jobs[run_id]["zip_path"] = str(zip_path)
        _save_jobs(jobs)

    except Exception as e:
        jobs = _load_jobs()
        jobs[run_id]["status"] = "error"
        jobs[run_id]["error"] = f"{e}\n{traceback.format_exc()}"
        _save_jobs(jobs)


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health", response_class=JSONResponse, tags=["default"])
def health():
    return {"ok": True}

@app.get("/", response_class=PlainTextResponse, tags=["default"])
def root():
    return "PreSeeds • Uniformidad API (OK)"

# Mini UI para subir imágenes (drag & drop) sin Swagger
@app.get("/app", response_class=HTMLResponse, tags=["default"])
def upload_ui():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>PreSeeds • Subir imágenes</title>
      <style>
        body { font-family: system-ui, -apple-system, Arial; padding: 24px; }
        .box { border: 2px dashed #999; padding: 24px; border-radius: 12px; }
        input, button { padding: 8px 12px; margin: 6px 0; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      </style>
    </head>
    <body>
      <h2>PreSeeds • Subir imágenes</h2>
      <form id="f" class="box">
        <div class="row">
          <div><label>Asesor: <input name="assessor" value="gonzalo"></label></div>
          <div><label>Campo: <input name="campo" value="campo 1"></label></div>
          <div><label>Cultivo: <input name="cultivo" value="trigo"></label></div>
          <div><label>Lote: <input name="lote" value="lote 1"></label></div>
          <div><label>Localidad: <input name="localidad" value="rosario"></label></div>
          <div><label>Área (m² por imagen): <input name="area_img_m2" value="26" type="number"></label></div>
          <div><label>Blocks: <input name="blocks" value="10" type="number"></label></div>
        </div>
        <p><input type="file" name="images" accept="image/jpeg,image/png" multiple></p>
        <button type="submit">Crear job</button>
      </form>

      <pre id="out"></pre>

      <script>
        const f = document.getElementById('f');
        const out = document.getElementById('out');
        f.onsubmit = async (e) => {
          e.preventDefault();
          const fd = new FormData(f);
          out.textContent = 'Subiendo...';
          const r = await fetch('/jobs', { method: 'POST', body: fd });
          const j = await r.json();
          out.textContent = JSON.stringify(j, null, 2) + "\\n\\n" +
            "Status: /status/" + j.run_id + "\\n" +
            "Download: /download/" + j.run_id;
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/jobs", tags=["default"])
async def create_job(
    background: BackgroundTasks,
    images: List[UploadFile] = File(..., description="Una o varias imágenes JPG/PNG"),
    assessor: str = Form(""),
    campo: str = Form(""),
    cultivo: str = Form(""),
    lote: str = Form(""),
    localidad: str = Form(""),
    area_img_m2: int = Form(26),
    blocks: int = Form(10),
):
    """
    Crea un job: guarda las imágenes en /tmp/<run_id>/input_* y dispara el procesamiento en background.
    Retorna inmediatamente un run_id.
    """
    # Validación rápida: solo aceptamos JPG/PNG
    valid_types = {"image/jpeg", "image/png"}
    for f in images:
        if f.content_type not in valid_types:
            raise HTTPException(status_code=415, detail=f"Tipo no soportado: {f.content_type}")

    run_id = uuid4().hex
    work_dir = TMP_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for idx, up in enumerate(images, start=1):
        data = await up.read()
        if len(data) == 0:
            continue
        # Nombre estándar
        ext = ".jpg" if up.filename.lower().endswith(".jpg") or up.content_type == "image/jpeg" else ".png"
        out_path = work_dir / f"input_{idx:03d}{ext}"
        with open(out_path, "wb") as f:
            f.write(data)
        saved_paths.append(out_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No se recibieron imágenes válidas.")

    meta = {
        "assessor": assessor,
        "campo": campo,
        "cultivo": cultivo,
        "lote": lote,
        "localidad": localidad,
        "area_img_m2": area_img_m2,
    }

    # Guardamos job en "DB" y disparamos background
    jobs = _load_jobs()
    jobs[run_id] = {
        "status": "pending",
        "created_at": time.time(),
        "meta": meta,
        "zip_path": "",
        "error": "",
    }
    _save_jobs(jobs)

    background.add_task(process_images, run_id, saved_paths, blocks, meta)

    return {"run_id": run_id, "status": "pending"}


@app.get("/status/{run_id}", tags=["default"])
def job_status(run_id: str):
    jobs = _load_jobs()
    if run_id not in jobs:
        raise HTTPException(status_code=404, detail="run_id no encontrado")
    job = jobs[run_id]
    return {"run_id": run_id, "status": job["status"], "error": job.get("error", "")}


@app.get("/download/{run_id}", response_class=FileResponse, tags=["default"])
def job_download(run_id: str):
    jobs = _load_jobs()
    if run_id not in jobs:
        raise HTTPException(status_code=404, detail="run_id no encontrado")
    job = jobs[run_id]
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job no listo. Estado: {job['status']}")
    zip_path = Path(job["zip_path"])
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="ZIP no encontrado")
    return FileResponse(str(zip_path), filename=f"{run_id}.zip", media_type="application/zip")
