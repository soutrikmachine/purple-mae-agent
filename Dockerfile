FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=9009

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY *.py ./
COPY amber-manifest.json5 agent.toml ./

EXPOSE 9009

# Default: pure-strategy mode. USE_LLM=false, USE_RL=false, no external deps.
# Override at runtime to enable optional refinement layers:
#   docker run -e USE_LLM=true -e OPENROUTER_API_KEY=sk-or-... ...
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "9009"]
