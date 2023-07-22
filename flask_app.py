from flask import Flask
from endpoints import model_routes
from flask_cors import CORS
import os
from threading import Thread
from azure.storage.blob import BlobServiceClient
import tensorflow as tf
import joblib
from appconfig import get_all_configurations
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#Azure app config
AZURE_CONFIG_CONNECTION_STRING = "Endpoint=https://app-config-predictive.azconfig.io;Id=K39n;Secret=Z0oNxiCBtKfIk90Gj2Yftdlv85XPD76uL1/sGCLwy1k="
app.config.update(get_all_configurations(AZURE_CONFIG_CONNECTION_STRING))

#Azure app insights
INSTRUMENTATION_KEY = app.config.get("instrument_key_appinsights")
print('INSTRUMENTATION_KEY',INSTRUMENTATION_KEY)



app.register_blueprint(model_routes)
download_completed = False

# logger = create_custom_logger(app)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=INSTRUMENTATION_KEY))
logger.warning('logging added')
app.logger = logger


# Function to download the model from Azure Blob Storage
def download_model(connection_string, container_name):
    try:
        global download_completed
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        download_directory = app.config['files']['model_filename']
        download_file = app.config['files']['scaler_filename']
        os.makedirs(download_directory, exist_ok=True)

        blobs = container_client.list_blobs(name_starts_with=download_directory)
        for blob in blobs:
            local_path = os.path.join('.', blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as file:
                file.write(container_client.get_blob_client(blob).download_blob().readall())

        # Download the specified file
        blob_client = container_client.get_blob_client(download_file)
        with open(download_file, "wb") as file:
            file.write(blob_client.download_blob().readall())

        loaded_model = tf.keras.models.load_model("./" + download_directory)
        scaler = joblib.load("./" + download_file)

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
        logger.info("flask app started successfully on port 5000")
        app.run(host='0.0.0.0', port=5000)

    else:
        logger.error("Model download failed. Flask app will not be started.")
        print("Model download failed. Flask app will not be started.")
