import os
import shutil
import uuid
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional

# Inicializar API
app = FastAPI(
    title="PreSeeds · Uniformidad API",
    version="0.2.0",
    description="API para procesar imágenes de uniformidad de siembra."
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta temporal para procesamiento
TEMP_DIR = "temp_uploads"
RESULTS_DIR = "results"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de Uniformidad PreSeeds"}

@app.post("/uniformidad/infer")
async def uniformidad_infer(
    assessor: str = Form(...),
    campo: str = Form(...),
    cultivo: str = Form(...),
    lote: str = Form(...),
    localidad: str = Form(...),
    area_img_m2: float = Form(...),
    blocks: int = Form(...),
    images: Optional[List[UploadFile]] = File(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Procesa imágenes de uniformidad de siembra.
    Se aceptan archivos en `images` o `files`.
    """
    try:
        # Unificar lista de archivos
        uploaded_files = images or files
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No se han subido imágenes.")

        # Crear ID de ejecución
        run_id = str(uuid.uuid4())
        run_dir = os.path.join(TEMP_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Guardar imágenes
        saved_paths = []
        for img in uploaded_files:
            file_path = os.path.join(run_dir, img.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(img.file, f)
            saved_paths.append(file_path)

        # Simulación de procesamiento (aquí iría tu pipeline real)
        output_zip = os.path.join(RESULTS_DIR, f"{run_id}.zip")
        shutil.make_archive(output_zip.replace(".zip", ""), 'zip', run_dir)

        return {"run_id": run_id, "archivos_recibidos": len(saved_paths)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imágenes: {str(e)}")

@app.get("/download/{run_id}")
async def download(run_id: str):
    """Descarga el ZIP de resultados."""
    file_path = os.path.join(RESULTS_DIR, f"{run_id}.zip")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return FileResponse(file_path, filename=f"uniformidad_{run_id}.zip", media_type="application/zip")
