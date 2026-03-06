# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./backend/requirements.txt .

# Install any needed packages specified in requirements.txt
# Also install build-essential for packages that need to compile C code
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    apt-get purge -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the content of the backend directory to the working directory
COPY ./backend/ .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables (can be overridden)
ENV NEO4J_URI="bolt://neo4j:7687"
ENV API_KEY="default_key"

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
