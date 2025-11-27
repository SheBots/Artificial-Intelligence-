FROM python:3.10.12-slim

# Prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies for lxml, pillow, faiss
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose port for Render
EXPOSE 8001

# Start command
CMD ["uvicorn", "rag.rag_main:app", "--host", "0.0.0.0", "--port", "8001"]
