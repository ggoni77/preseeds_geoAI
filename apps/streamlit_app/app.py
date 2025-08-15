import os
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title='PreSeeds · Uniformidad', layout='wide')
st.title('PreSeeds · Uniformidad de Siembra (MVP)')

# URL de la API: env var (Render) -> secrets -> localhost
api_url = os.getenv('API_URL') or st.secrets.get('API_URL', 'http://localhost:8000')

with st.sidebar:
    st.header('Datos del lote')
    asesor = st.text_input('Asesor', '')
    campo = st.text_input('Campo', '')
    cultivo = st.text_input('Cultivo', '')
    lote = st.text_input('Lote', '')
    localidad = st.text_input('Localidad', '')
    blocks = st.number_input('Bloques (10x10 por defecto)', min_value=5, max_value=20, value=10)

uploaded = st.file_uploader(
    'Subí una o varias imágenes RGB (JPG/PNG)', 
    accept_multiple_files=True, type=['jpg','jpeg','png']
)

if st.button('Procesar') and uploaded:
    with st.spinner('Procesando...'):
        files = [('images', (u.name, u.getvalue(), 'image/jpeg')) for u in uploaded]
        data = {
            'asesor': asesor, 'campo': campo, 'cultivo': cultivo,
            'lote': lote, 'localidad': localidad, 'blocks': str(blocks)
        }
        try:
            r = requests.post(f"{api_url}/uniformidad/infer", files=files, data=data, timeout=180)
            if r.status_code == 200:
                out = r.json()
                st.success(out['message'])
                dl = f"{api_url}/download/{out['run_id']}"
                st.markdown(f"**Descargar resultados:** [ZIP]({dl})")
            else:
                st.error(f"Error: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"No pude conectar con la API: {e}")

st.caption('Tip: en producción, seteá la variable de entorno API_URL para apuntar a tu servicio FastAPI.')
