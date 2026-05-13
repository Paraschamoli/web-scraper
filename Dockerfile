# Use official Python runtime as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install system dependencies for Playwright browsers and common tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pyproject.toml
WORKDIR /app
COPY pyproject.toml ./

# Install pip and the project dependencies (without the project itself, for caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Install Playwright and its Chromium browser
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy the rest of the application code
COPY . .

# Default command: show help for crawlee_scraper (adjust if needed)
ENTRYPOINT ["python", "crawlee_scraper.py"]
CMD ["--help"]