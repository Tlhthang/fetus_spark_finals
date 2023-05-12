# Use the base Python image from Docker
FROM python:3.7

# Install Java and Spark dependencies
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean;

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make the default port available outside this container
EXPOSE 8000

CMD ["python", "Spark_FastAPI.py"]
