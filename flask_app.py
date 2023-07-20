from flask import Flask
import yaml
from endpoints import model_routes
from flask_cors import CORS
import os
from threading import Thread
from azure.storage.blob import BlobServiceClient
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Read YAML configuration file
with open('config/config_model.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set Flask configuration variables
app.config.update(config)
app.register_blueprint(model_routes)

# Flag to track model download status
download_completed = False

# Function to download the model from Azure Blob Storage
def download_model(connection_string, container_name):
    global download_completed

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        blobs = container_client.list_blobs()

        for blob in blobs:
            # Create the local directory structure based on the blob's directory path
            blob_client = container_client.get_blob_client(blob)
            local_path = os.path.join(".", blob.name)
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)

            with open(local_path, "wb") as file:
                file.write(blob_client.download_blob().readall())

            loaded_model = tf.keras.models.load_model(app.config['files']['model_filename'])
            scaler = joblib.load(app.config['files']['scaler_filename'])

        # Set download_completed flag to True when the download is successful
        download_completed = True
        with app.app_context():
            app.loaded_model = loaded_model
            app.scaler = scaler

    except Exception as e:
        print("Error while downloading the model:", str(e))
        download_completed = False

if __name__ == '__main__':
    # Download the model in a separate thread
    download_thread = Thread(target=download_model, args=(app.config['connection_string'], app.config['container_name']))
    download_thread.start()

    # Wait until the download is completed
    download_thread.join()

    # Start the Flask app only if the download was successful
    if download_completed:
        # Attach the loaded model and scaler to current_app
        app.run(host='0.0.0.0', port=5000)
    else:
        print("Model download failed. Flask app will not be started.")
