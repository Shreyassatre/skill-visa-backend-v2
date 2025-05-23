# Dockerfile

# Use an official lightweight Python image
# python:3.10-slim is a good balance of features and size
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install the dependencies
# Use --no-cache-dir to avoid storing cache, reducing image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and data file into the working directory
COPY main.py .
COPY aus_visa_occupations_with_embeddings.json .

# EXPOSE the port the application will listen on
# The PORT environment variable is commonly used by hosting platforms
# We expose 8000 as a convention, but the app inside will listen on $PORT
EXPOSE 8000

# Define the command to run the application
# This overrides the default Python entrypoint
# We use uvicorn directly, binding to 0.0.0.0 (accessible externally from container)
# and reading the PORT environment variable.
# Using sh -c allows shell expansion of $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Optional: Define a default value for PORT if the hosting platform doesn't set it
# This is less common for cloud hosts, but can be useful.
# ENV PORT=8000 # You can uncomment this if needed, but CMD handles default too