# from azure.appconfiguration import AzureAppConfigurationClient
import yaml


# def get_all_configurations(connection_string):
def get_all_configurations():

    # config_client = AzureAppConfigurationClient.from_connection_string(connection_string)

    # Get all configurations from Azure Config Service
    # configurations = config_client.list_configuration_settings()

    with open('config/config_model.yml', 'r') as config_file:
        configurations = yaml.safe_load(config_file)

    return configurations

    #
    # # Create a dictionary to store the configurations
    # config_dict = {}
    #
    # # Loop through all configurations and store them in the dictionary
    # for config in configurations:
    #     # Split the key into parts based on the separator (in this case, ':')
    #     key_parts = config.key.split('.')
    #
    #     # Initialize the current_dict as the main dictionary
    #     current_dict = config_dict
    #
    #     # Loop through key parts to create nested dictionaries
    #     for i, key_part in enumerate(key_parts):
    #         if i == len(key_parts) - 1:
    #             # For the last part, store the value
    #             current_dict[key_part] = config.value
    #         else:
    #             # If the nested key doesn't exist, create an empty dictionary for it
    #             current_dict = current_dict.setdefault(key_part, {})
    #
    # return config_dict
if __name__ == "__main__":
    all_configurations = get_all_configurations()
    # print(all_configurations)
