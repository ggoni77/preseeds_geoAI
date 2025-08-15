# PreSeeds · GeoAI (MVP Uniformidad)

## Local
API:
  cd apps/fastapi_service && uvicorn main:app --reload
UI:
  cd apps/streamlit_app && streamlit run app.py

## Health
GET /health → {"ok": true}

## Deploy (Render)
API: infra/Dockerfile.api  (port 8000)
UI : infra/Dockerfile.ui   (port 8501, env API_URL = <URL de la API>)
