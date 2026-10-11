FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers
COPY supportsense ./supportsense
COPY service ./service
COPY streaming ./streaming
COPY dashboard ./dashboard
COPY training ./training
EXPOSE 8000
CMD ["uvicorn", "service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
