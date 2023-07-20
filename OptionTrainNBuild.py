import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import Adam
import joblib
import logging
import tensorflow as tf


def load_dataset(filename):
    try:
        df = pd.read_csv(filename)
        df['Maturity'] = df['Maturity'] / 365
        df['Spot Price'] = df['Spot Price'] / df['Strike Price']
        df['Call_Premium'] = df['Call_Premium'] / df['Strike Price']
        X = df.drop('Call_Premium', axis=1)
        Y = df['Call_Premium']
        return X, Y
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while loading the dataset: %s", str(e))


def scale_data(X_train, X_test):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while scaling the data: %s", str(e))


def save_model(model, scaler, model_filename, scaler_filename):
    try:
        # Save the model and scaler
        model.save(model_filename, save_format='tf')
        joblib.dump(scaler, scaler_filename)
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while saving the model and scaler: %s", str(e))


def build_model(input_dim):
    try:
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim))
        model.add(Activation('elu'))
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64))
        model.add(Activation('elu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)
        return model
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while building the model: %s", str(e))


def model_custom_predict(Spot_Price, Strike_Price, Maturity, risk_free_interest, Volatility, model_filename,
                         scaler_filename):
    try:
        inputs_to_model = pd.DataFrame({'Spot Price': [Spot_Price / Strike_Price],
                                        'Strike Price': [Strike_Price],
                                        'Maturity': [Maturity / 365],
                                        'risk_free_interest': [risk_free_interest],
                                        'Volatility': [Volatility]
                                        })

        loaded_model = tf.keras.models.load_model(model_filename)
        scaler = joblib.load(scaler_filename)
        input_data_scaled = scaler.transform(inputs_to_model)
        value = loaded_model.predict(input_data_scaled)
        option_value = value * Strike_Price
        return option_value
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while predicting the option value: %s", str(e))

import numpy as np
def model_custom_predict_multiple(json_data, loaded_model, scaler):
    try:
        # Parse the JSON input data
        data = json_data

        # Get the list of records from the JSON data
        records = data['records']
        # loaded_model = tf.keras.models.load_model(model_filename)
        # scaler = joblib.load(scaler_filename)

        results_df = pd.DataFrame(
            columns=["Spot_Price", "Strike_Price", "Maturity", "risk_free_interest", "Volatility", "Call_Premium",
                     "Option_Value"])

        # Extract the input variables from the 'records' list of dictionaries
        spot_prices = np.array([record['Spot_Price'] for record in records])
        strike_prices = np.array([record['Strike_Price'] for record in records])
        maturities = np.array([record['Maturity'] for record in records])
        risk_free_interests = np.array([record['risk_free_interest'] for record in records])
        volatilities = np.array([record['Volatility'] for record in records])
        cal_val_mcs = np.array([record['Call_Premium'] for record in records])

        # Perform the prediction for all records at once
        inputs_to_model = pd.DataFrame({
            'Spot Price': spot_prices / strike_prices,
            'Strike Price': strike_prices,
            'Maturity': maturities / 365,
            'risk_free_interest': risk_free_interests,
            'Volatility': volatilities
        })

        # Scale the input data (assuming 'scaler' is the pre-fitted StandardScaler)
        input_data_scaled = pd.DataFrame(scaler.transform(inputs_to_model.values), columns=inputs_to_model.columns)

        # Predict the option values for all records at once
        values = loaded_model.predict(input_data_scaled)
        option_values = values * strike_prices

        # Construct the result DataFrame
        results_df["Spot_Price"] = spot_prices
        results_df["Strike_Price"] = strike_prices
        results_df["Maturity"] = maturities
        results_df["risk_free_interest"] = risk_free_interests
        results_df["Volatility"] = volatilities
        results_df["Call_Premium"] = cal_val_mcs
        results_df["Option_Value"] = np.round(option_values, 2)

        # Convert the result DataFrame to a list of dictionaries
        results = results_df.to_dict(orient='records')
        return  results
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while predicting the option values: %s", str(e))
# def model_custom_predict_multiple(json_data, model_filename, scaler_filename):
#     try:
#         # Parse the JSON input data
#         data = json_data
#
#         # Get the list of records from the JSON data
#         records = data['records']
#
#         # Load the model and scaler
#         loaded_model = tf.keras.models.load_model(model_filename)
#         scaler = joblib.load(scaler_filename)
#
#         # Convert records to a pandas DataFrame for easier processing
#         df = pd.DataFrame(records)
#
#         # Rename columns to match the expected output format
#         df.rename(columns={'Spot_Price': 'Spot Price', 'Strike_Price': 'Strike Price', 'Maturity': 'Maturity',
#                            'risk_free_interest': 'risk_free_interest', 'Volatility': 'Volatility',
#                            'Call_Premium': 'Call_Premium'}, inplace=True)
#
#         # Scale the input data
#         df['Spot Price'] /= df['Strike Price']
#         df['Maturity'] /= 365
#
#         # Select the columns needed for prediction
#         prediction_columns = ['Spot Price', 'Strike Price', 'Maturity', 'risk_free_interest', 'Volatility']
#         inputs_to_model = df[prediction_columns]
#
#         # Scale the input data (make sure it's 1-dimensional)
#         input_data_scaled = scaler.transform(inputs_to_model.values)
#
#         # Predict the option values for all records at once
#         values = loaded_model.predict(input_data_scaled)
#
#         # Calculate option values and construct the result DataFrame
#         df['Option_Value'] = (values * df['Strike Price']).round(2)
#
#         # Convert the result DataFrame to a list of dictionaries
#         results = df.to_dict(orient='records')
#
#         return results
#     except Exception as e:
#         print('error', str(e))
#         logging.error("An error occurred while predicting the option values: %s", str(e))
#
#
#
#
# def model_custom_predict_multiple(json_data, model_filename, scaler_filename):
#     try:
#         # Parse the JSON input data
#         data = json_data
#
#         # Get the list of records from the JSON data
#         records = data['records']
#         loaded_model = tf.keras.models.load_model(model_filename)
#         scaler = joblib.load(scaler_filename)
#
#         # Initialize the list to store the results
#         results = []
#
#         for record in records:
#             # Extract the input variables for each record
#             spot_price = record['Spot_Price']
#             strike_price = record['Strike_Price']
#             maturity = record['Maturity']
#             risk_free_interest = record['risk_free_interest']
#             volatility = record['Volatility']
#             cal_val_mc = record['Call_Premium']
#
#             # Perform the prediction for each record
#             inputs_to_model = pd.DataFrame({
#                 'Spot Price': [spot_price / strike_price],
#                 'Strike Price': [strike_price],
#                 'Maturity': [maturity / 365],
#                 'risk_free_interest': [risk_free_interest],
#                 'Volatility': [volatility]
#             })
#
#             # Scale the input data
#             input_data_scaled = scaler.transform(inputs_to_model)
#
#             # Predict the option value for the record
#             value = loaded_model.predict(input_data_scaled)
#             option_value = value * strike_price
#
#             # Construct the result object with the expected order of keys
#             result = {
#                 "Spot_Price": spot_price,
#                 "Strike_Price": strike_price,
#                 "Maturity": maturity,
#                 "risk_free_interest": risk_free_interest,
#                 "Volatility": volatility,
#                 "Call_Premium":cal_val_mc,
#                 "Option_Value": round(option_value.item(),2)
#             }
#
#             # Append the result to the list
#             results.append(result)
#
#         return results
#     except Exception as e:
#         print('error', str(e))
#         logging.error("An error occurred while predicting the option values: %s", str(e))


def read1krecords(file_name):
    try:
        df = pd.read_csv(file_name)
        data = df.to_dict(orient='records')
        return data
    except Exception as e:
        print(str(e))
        logging.error("An error occurred while reading 1k records: %s", str(e))
