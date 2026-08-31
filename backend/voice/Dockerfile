FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the `voice` package's own files (not tests/README/env/etc.)
# into a `voice/` subdirectory, so `import voice.app` resolves against
# WORKDIR /app the same way it does locally against backend/ (the parent
# of backend/voice/).
COPY __init__.py app.py audio.py config.py fillers.py models.py observability.py prefetch.py sessions.py turn_controller.py ./voice/
COPY adapters ./voice/adapters
COPY clients ./voice/clients

RUN useradd -m -u 1000 voiceservice
USER voiceservice

EXPOSE 8090

CMD ["uvicorn", "voice.app:app", "--host", "0.0.0.0", "--port", "8090"]
