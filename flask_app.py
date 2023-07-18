from flask import Flask
import yaml
from endpoints import model_routes


app = Flask(__name__)

# Read YAML configuration file
with open('config/config_model.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set Flask configuration variables
app.config.update(config)
app.register_blueprint(model_routes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
