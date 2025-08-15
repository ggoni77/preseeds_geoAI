from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pandas as pd
import matplotlib.pyplot as plt
import io

# Gráfico de barras (Baja/Media/Alta)
def plot_barras(clases_count: dict) -> bytes:
    fig, ax = plt.subplots()
    labels = list(clases_count.keys())
    values = [clases_count[k] for k in labels]
    ax.bar(labels, values)
    ax.set_ylabel('Bloques (#)')
    ax.set_title('Distribución de cobertura por bloques')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# Crea PDF con imagen original/procesada y resumen
def build_pdf(pdf_path: str, titulo: str, info: dict, img_path: str, proc_png_bytes: bytes, barras_png_bytes: bytes):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Encabezado
    c.setFont('Helvetica-Bold', 16)
    c.drawString(40, h-50, titulo)
    c.setFont('Helvetica', 10)
    c.drawString(40, h-70, f"Asesor: {info.get('asesor','')}  |  Campo: {info.get('campo','')}  |  Cultivo: {info.get('cultivo','')}")
    c.drawString(40, h-85, f"Lote: {info.get('lote','')}  |  Localidad: {info.get('localidad','')}")

    # Imagen original y procesada
    y_img = h-350
    try:
        c.drawImage(ImageReader(img_path), 40, y_img, width=250, height=250, preserveAspectRatio=True, mask='auto')
    except:
        pass
    c.drawImage(ImageReader(io.BytesIO(proc_png_bytes)), 310, y_img, width=250, height=250, preserveAspectRatio=True, mask='auto')

    # Resumen
    c.setFont('Helvetica-Bold', 12)
    c.drawString(40, y_img-20, 'Resumen')
    c.setFont('Helvetica', 10)
    c.drawString(40, y_img-35, f"Cobertura estimada (mitad inferior): {info.get('pct_cobertura',0):.2f}%")
    c.drawString(40, y_img-50, f"Bloques Alta/Media/Baja: {info.get('alta',0)}/{info.get('media',0)}/{info.get('baja',0)}")

    # Gráfico de barras
    c.drawImage(ImageReader(io.BytesIO(barras_png_bytes)), 40, 60, width=520, height=220, preserveAspectRatio=True, mask='auto')

    c.showPage()
    c.save()

# Exporta CSV de stats por bloque
def export_csv(csv_path: str, stats):
    df = pd.DataFrame(stats)
    df.to_csv(csv_path, index=False)
    return csv_path
