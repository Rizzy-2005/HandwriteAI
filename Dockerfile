# -----------------------------------------------------------------
# STAGE 1: Define the base environment
# -----------------------------------------------------------------
# Use an official, lightweight Python image. Using a specific version is a best practice.
FROM python:3.9-slim

# -----------------------------------------------------------------
# STAGE 2: Set up the working environment and install dependencies
# -----------------------------------------------------------------
# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
# This is done separately to leverage Docker's layer caching.
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt
# --no-cache-dir makes the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------
# STAGE 3: Copy application code and set run command
# -----------------------------------------------------------------
# Copy the rest of your application's source code into the container at /app
COPY . .

# Expose the port that Gunicorn will run on. This tells Docker the container is listening on this port.
EXPOSE 8000

# The command that will be executed when the container starts.
# This runs your Flask app using the Gunicorn production server.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers=4", "app:app"]