FROM python:3.11-slim

WORKDIR /app

# Reuse the same requirements (it includes streamlit + requests).
COPY backend/requirements.txt .
RUN pip install --no-cache-dir streamlit==1.40.2 requests==2.32.3

COPY frontend/ ./frontend/

WORKDIR /app/frontend

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
