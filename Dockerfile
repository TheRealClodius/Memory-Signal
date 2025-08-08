FROM ghcr.io/astral-sh/uv:python3.11-bookworm

# System deps (bookworm)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the mcp server folder
COPY memoryos-mcp/ ./memoryos-mcp/

# Create venv and install deps using uv
WORKDIR /app/memoryos-mcp
RUN uv venv .venv && \
    . .venv/bin/activate && \
    awk '{ if ($0 ~ /^faiss-gpu/) { print "faiss-cpu>=1.7.0,<2.0.0" } else { print $0 } }' requirements.txt > /tmp/requirements.patched.txt && \
    uv pip install -r /tmp/requirements.patched.txt

# Copy .env.local at runtime via Railway variables; not baked into image

# Expose port for HTTP transport
ENV HOST=0.0.0.0
ENV PORT=8000
ENV TRANSPORT=http

# Default command: run HTTP server
CMD . .venv/bin/activate && python server_new.py --config config.json --transport ${TRANSPORT} --host ${HOST} --port ${PORT}