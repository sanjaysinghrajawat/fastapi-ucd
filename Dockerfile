# Base image
FROM python:3.10-slim

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/cache

# Set working directory
WORKDIR /app

# Create cache directory
RUN mkdir -p /app/cache

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('google/flan-t5-large'); \
    AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')"



# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
