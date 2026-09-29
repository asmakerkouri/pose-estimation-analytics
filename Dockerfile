FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image small
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY posekit ./posekit
# bake the weights into the image (no download at start-up)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')" && chown -R app /app
USER app
ENV POSE_WEIGHTS=/app/yolo11n-pose.pt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "posekit.api:app", "--host", "0.0.0.0", "--port", "8000"]
