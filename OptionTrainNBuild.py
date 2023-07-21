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


def model_custom_predict(Spot_Price, Strike_Price, Maturity, risk_free_interest, Volatility, loaded_model,
                         scaler):
    try:
        inputs_to_model = pd.DataFrame({'Spot Price': [Spot_Price / Strike_Price],
                                        'Strike Price': [Strike_Price],
                                        'Maturity': [Maturity / 365],
                                        'risk_free_interest': [risk_free_interest],
                                        'Volatility': [Volatility]
                                        })

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
        data = json_data

        records = data['records']
        batch_size = 100
        record_chunks = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]

        results = []

        for chunk in record_chunks:
            # Extract the input variables for each record in the chunk
            spot_prices = np.array([record['Spot_Price'] for record in chunk])
            strike_prices = np.array([record['Strike_Price'] for record in chunk])
            maturities = np.array([record['Maturity'] for record in chunk])
            risk_free_interests = np.array([record['risk_free_interest'] for record in chunk])
            volatilities = np.array([record['Volatility'] for record in chunk])

            # Perform the prediction for the chunk
            inputs_to_model = pd.DataFrame({
                'Spot Price': np.divide(spot_prices, strike_prices),
                'Strike Price': strike_prices,
                'Maturity': np.divide(maturities, 365),
                'risk_free_interest': risk_free_interests,
                'Volatility': volatilities
            })

            # Scale the input data
            input_data_scaled = scaler.transform(inputs_to_model)

            # Predict the option values for the chunk
            values = loaded_model.predict(input_data_scaled)
            option_values = values.flatten() * strike_prices

            # Construct the result objects for the chunk
            for i, record in enumerate(chunk):
                result = {
                    "Spot_Price": spot_prices[i].item(),
                    "Strike_Price": strike_prices[i].item(),
                    "Maturity": maturities[i].item(),
                    "risk_free_interest": risk_free_interests[i].item(),
                    "Volatility": volatilities[i].item(),
                    "Option_Value": round(option_values[i].item(), 2),
                    "Call_Premium": record["Call_Premium"]
                }
                results.append(result)
            print(results)
            return results

    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while predicting the option values: %s", str(e))

def read1krecords(file_name):
    try:
        df = pd.read_csv(file_name)
        data = df.to_dict(orient='records')
        return data
    except Exception as e:
        print(str(e))
        logging.error("An error occurred while reading 1k records: %s", str(e))
