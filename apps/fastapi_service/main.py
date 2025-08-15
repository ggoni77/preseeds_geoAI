import os, zipfile, uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd

# Imports absolutos (sin el punto) para evitar errores de relative import
from inference import infer_image
from pdf_report import build_pdf, export_csv, plot_barras

app = FastAPI(title='PreSeeds · Uniformidad API')

# Carpeta de salida para resultados
OUT_DIR = os.environ.get('OUT_DIR', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

@app.post('/uniformidad/infer')
async def uniformidad_infer(
    images: list[UploadFile] = File(...),
    asesor: str = Form(''), campo: str = Form(''), cultivo: str = Form(''),
    lote: str = Form(''), localidad: str = Form(''), area_img_m2: float = Form(26.0), blocks: int = Form(10)
):
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(OUT_DIR, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)

    all_rows = []
    zippath = os.path.join(run_dir, 'resultados.zip')

    with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for idx, up in enumerate(images):
            img_bytes = await up.read()
            res = infer_image(img_bytes, blocks=blocks)

            # CSV por imagen
            csv_path = os.path.join(run_dir, f'stats_{idx+1}.csv')
            export_csv(csv_path, res['blocks_stats'])
            zf.write(csv_path, os.path.basename(csv_path))

            # Conteo de clases para el gráfico de barras
            classes = pd.DataFrame(res['blocks_stats'])['clase'].value_counts().to_dict()
            alta = classes.get('Alta', 0)
            media = classes.get('Media', 0)
            baja = classes.get('Baja', 0)

            # PDF por imagen
            pdf_path = os.path.join(run_dir, f'reporte_{idx+1}.pdf')
            meta = {
                'asesor': asesor,
                'campo': campo,
                'cultivo': cultivo,
                'lote': lote,
                'localidad': localidad,
                'pct_cobertura': res['pct_global'],
                'alta': alta,
                'media': media,
                'baja': baja
            }
            orig_path = os.path.join(run_dir, f'orig_{idx+1}.jpg')
            with open(orig_path, 'wb') as f:
                f.write(img_bytes)
            barras_png = plot_barras({'Baja': baja, 'Media': media, 'Alta': alta})
            build_pdf(pdf_path, 'Informe de Uniformidad de Siembra', meta, orig_path, res['processed_png'], barras_png)
            zf.write(pdf_path, os.path.basename(pdf_path))

            # Guardar imagen procesada PNG
            proc_path = os.path.join(run_dir, f'procesada_{idx+1}.png')
            with open(proc_path, 'wb') as f:
                f.write(res['processed_png'])
            zf.write(proc_path, os.path.basename(proc_path))

            all_rows.append({
                'imagen': up.filename,
                'pct_cobertura': res['pct_global'],
                'alta': alta,
                'media': media,
                'baja': baja
            })

        # CSV global con resumen de todas las imágenes
        global_csv = os.path.join(run_dir, 'resumen_global.csv')
        pd.DataFrame(all_rows).to_csv(global_csv, index=False)
        zf.write(global_csv, os.path.basename(global_csv))

    return JSONResponse({
        'run_id': run_id,
        'zip_path': zippath,
        'message': f'Procesamiento OK. {len(images)} imagen(es).'
    })


@app.get('/download/{run_id}')
def download_zip(run_id: str):
    zippath = os.path.join(OUT_DIR, f'run_{run_id}', 'resultados.zip')
    if not os.path.exists(zippath):
        return JSONResponse({'error': 'No encontrado'}, status_code=404)
    return FileResponse(zippath, media_type='application/zip', filename=f'preseeds_uniformidad_{run_id}.zip')
