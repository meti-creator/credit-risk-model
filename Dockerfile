# Use an official lightweight Python runtime environment image
FROM python:3.12-slim

# Set the active workspace directory inside the container capsule
WORKDIR /app

# Copy your local dependencies file first to leverage Docker's caching layer
COPY requirements.txt .

# Install the required libraries inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container workspace
COPY . .

# Open up port 8000 inside the container environment
EXPOSE 8000

# Tell the container exactly how to boot up your FastAPI server using Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]