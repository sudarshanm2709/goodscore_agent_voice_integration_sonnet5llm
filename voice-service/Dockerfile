FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY voice_service ./voice_service

RUN useradd -m -u 1000 voiceservice
USER voiceservice

EXPOSE 8090

CMD ["uvicorn", "voice_service.app:app", "--host", "0.0.0.0", "--port", "8090"]
