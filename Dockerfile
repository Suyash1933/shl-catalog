FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (avoids downloading 1GB+ CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the FAISS index at build time
RUN python -c "from retrieval import build_index; build_index()"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
