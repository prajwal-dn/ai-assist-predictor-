FROM python:3.10-slim

# Create user to run the app (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Hugging Face exposes port 7860 by default
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
