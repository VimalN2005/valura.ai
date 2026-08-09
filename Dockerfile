# Use a slim Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY agents/ ./agents/
COPY harness/ ./harness/
COPY schema/ ./schema/
COPY tools/ ./tools/
# Create data folder placeholder
RUN mkdir -p data

# Set the default entrypoint to run the harness loop
ENTRYPOINT ["python", "-m", "harness.run"]

# Default argument to run in qualifying (scored) mode when deployed
CMD ["--mode", "qualifying"]
