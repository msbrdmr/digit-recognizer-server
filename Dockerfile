# Use the official Python image as a base image
FROM python:3.10.5

# Set the working directory inside the container
WORKDIR /app/backend

# Copy the backend code into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the backend server will run on
EXPOSE 8000

# Command to run the backend server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
