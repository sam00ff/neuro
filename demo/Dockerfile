# NeuroLinked V1.3 — Public Demo Server Image
#
# Runs a rate-limited, ephemeral-tenant version of the brain for public
# demo. Any visitor can talk to it; each session gets its own tenant
# that gets wiped after idle timeout.

FROM python:3.11-slim

WORKDIR /app

# System deps for scipy / numpy / opencv wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ build-essential \
        libgl1 libglib2.0-0 \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir cryptography

COPY . .

# Demo-mode env vars
ENV NLKD_DEMO_MODE=1 \
    NLKD_DEMO_IDLE_TIMEOUT_SEC=900 \
    NLKD_DEMO_RATE_LIMIT_PER_MIN=30 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Brain listens on 0.0.0.0 inside the container
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000", "--neurons", "10000"]
