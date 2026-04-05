FROM python:3.9.18

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    samtools \
    wget \
    curl \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ────────────────────────────────────────────────────
WORKDIR /workspace

# ── Copy requirements (pinned) ───────────────────────────────────────────────
COPY requirements.lock.txt /workspace/requirements.lock.txt

# ── Install Python dependencies ─────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.lock.txt

# ── Copy project ─────────────────────────────────────────────────────────────
COPY . /workspace/

# ── Environment variable (optional) ─────────────────────────────────────────
ENV PYTHONPATH=/workspace/src

# ── Default command ─────────────────────────────────────────────────────────
CMD ["python", "-c", "from deepISA.utils.deepisa_guard import validate_deepisa_environment; validate_deepisa_environment(); print('Environment OK')"]
