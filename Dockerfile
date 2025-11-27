FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that your FastAPI app runs on
EXPOSE 8001

# Run the application
CMD ["uvicorn", "rag.rag_main:app", "--host", "0.0.0.0", "--port", "8001"]