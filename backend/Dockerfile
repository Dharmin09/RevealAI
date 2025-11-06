FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Copy project files
COPY . /app/

# Install dependencies from compatible requirements file
# This ensures all versions work together with TensorFlow 2.13
COPY requirements-compatible.txt /app/
RUN pip install --no-cache-dir -r requirements-compatible.txt -q

# Expose port
EXPOSE 5000

# Set environment variables for optimal TensorFlow performance
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1

# Run API in production mode
CMD ["python", "api_production_v2.py"]
