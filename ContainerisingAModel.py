# Step-by-step guide:
# Step 1: Save the trained model
# Start by saving your trained model in a format that can be easily loaded and used in production. Depending on the framework, this could be a .pkl file for Scikit-learn models, an .h5 file for Keras models, or a SavedModel format for TensorFlow.

#  Scikit-learn Example
import joblib
joblib.dump(model, 'model.pkl')

#TensorFlow Example
model.save('model.h5')

# Step 2: Define dependencies
# Create a requirements.txt file that lists all the Python libraries your model depends on. This ensures that the exact versions of the dependencies are installed in the production environment.

# Example requirements.txt  
numpy==1.21.2
pandas==1.3.3
scikit-learn==0.24.2
tensorflow==2.6.0

# Step 3: Include preprocessing and postprocessing code
# If your model requires specific data preprocessing steps (e.g., scaling, encoding) or post processing (e.g., thresholding), include these steps in your package. It’s best to encapsulate this logic in functions that can be easily called during inference.

# Example
def preprocess(input_data):
    # Example preprocessing steps
    scaled_data = scaler.transform(input_data)
    return scaled_data

def postprocess(predictions):
    # Example postprocessing steps
    return (predictions > 0.5).astype(int)

# Step 4: Package the model
# Use a packaging tool like setuptools to bundle your model, dependencies, and code into a distributable package.

# Example setup.py
from setuptools import setup, find_packages

setup(
    name='my_model_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.21.2',
        'pandas==1.3.3',
        'scikit-learn==0.24.2',
        'tensorflow==2.6.0'
    ],
    scripts=['scripts/preprocess.py', 'scripts/postprocess.py']
)






# Containerizing your model with Docker
# Docker is a popular tool for containerizing applications, including machine learning models. A Docker container bundles your model, dependencies, and environment into a portable and consistent unit. Here’s how to containerize your model:

# Step-by-step guide:
# Step 1: Create a Dockerfile
# A Dockerfile is a script that contains instructions to build a Docker image. It defines the base image, copies the necessary files, installs dependencies, and specifies the command to run the model.

# Example Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD [python, app.py]


# Step 2: Build the Docker image
# Once you have your Dockerfile, you can build your Docker image. This image will contain everything your model needs to run.

# Build the Docker image example
docker build -t my_model_image .


# Step 3: Run the Docker container
# After building the image, you can run it as a container. This container will behave the same way regardless of where it’s deployed, ensuring consistent performance across different environments.

# Run the Docker container example
docker run -d -p 80:80 my_model_image


# Step 4: Test the container locally
# Before deploying your container to a production environment, test it locally to ensure it’s functioning as expected. You can interact with your model via an API endpoint, typically using Flask or FastAPI.

# Example
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)