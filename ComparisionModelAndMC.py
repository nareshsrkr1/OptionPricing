import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import joblib

with open('config/config_model.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

model_filename = config['files']['model_filename']
scaler_filename = config['files']['scaler_filename']

loaded_model = tf.keras.models.load_model(model_filename)
scaler = joblib.load(scaler_filename)

csv_file = "InputDataSetLatest.csv"
df = pd.read_csv(csv_file)
df = df.iloc[:, :-1]
num_records = len(df)


# Initialize lists to store the time taken for each method
time_ml_predictions = []
time_mc_simulations = []


# Spot_Price,Strike_Price,Maturity,risk_free_interest,Volatility,Call_Premium
def simulate_option_price(row, num_simulations=100000):
    S = row['Spot Price']
    K = row['Strike Price']
    r = row['risk_free_interest']
    sigma = row['Volatility']
    T = row['Maturity'] / 365

    z = np.random.standard_normal((num_simulations,))

    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoffs = np.maximum(ST - K, 0)
    option_prices = payoffs * np.exp(-r * T)
    option_value = np.mean(option_prices)
    return option_value

def model_custom_predict_multiple(json_data, loaded_model, scaler, chunk_size=100):
    try:
        records = json_data

        # Initialize the results list to store the prediction results
        results = []

        # Split the records into chunks of the specified size
        record_chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

        for chunk in record_chunks:
            # Initialize lists to store the input variables for the chunk
            spot_prices = []
            strike_prices = []
            maturities = []
            risk_free_interests = []
            volatilities = []

            for record in chunk:
                # Extract the input variables for each record in the chunk
                spot_price = record['Spot Price']
                strike_price = record['Strike Price']
                maturity = record['Maturity']
                risk_free_interest = record['risk_free_interest']
                volatility = record['Volatility']

                # Append the input variables to the respective lists
                spot_prices.append(spot_price)
                strike_prices.append(strike_price)
                maturities.append(maturity)
                risk_free_interests.append(risk_free_interest)
                volatilities.append(volatility)

            # Perform the prediction for the chunk
            input_data = pd.DataFrame({
                'Spot Price': np.divide(spot_prices, strike_prices),
                'Strike Price': strike_prices,
                'Maturity': np.divide(maturities, 365),
                'risk_free_interest': risk_free_interests,
                'Volatility': volatilities
            })

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Predict the option values for the chunk
            values = loaded_model.predict(input_data_scaled, verbose=0)
            option_values = values.flatten() * np.array(strike_prices)

            # Construct the result objects for the chunk
            for i, record in enumerate(chunk):
                result = {
                    "Spot_Price": spot_prices[i],
                    "Strike_Price": strike_prices[i],
                    "Maturity": maturities[i],
                    "risk_free_interest": risk_free_interests[i],
                    "Volatility": volatilities[i],
                    "Option_Value": round(option_values[i], 2)
                    # "Call_Premium": record["Call_Premium"]
                }
                results.append(result)

        return results

    except Exception as e:
        error_msg = 'An error occurred while loading the dataset: ' + str(e)
        print('Error:', error_msg)
        return error_msg


# Compare the time for each row in the JSON data
start_time_mc = time.time()
for _, row in df.iterrows():
    mc_option_price = simulate_option_price(row)
end_time_mc = time.time()
time_mc_simulations.append(end_time_mc - start_time_mc)
print('end time mc',end_time_mc-start_time_mc)
# Convert the JSON data to a list of dictionaries
json_data = df.to_dict(orient='records')

# Compare the time for ML model predictions
start_time_ml = time.time()
ml_results = model_custom_predict_multiple(json_data, loaded_model, scaler)
end_time_ml = time.time()
time_ml_predictions.append(end_time_ml - start_time_ml)
print('end time ml',end_time_ml-start_time_ml)

# Calculate and print the average time taken for each method
avg_time_mc = np.mean(time_mc_simulations)
avg_time_ml = np.mean(time_ml_predictions)

# Plot the timings
labels = ['Monte Carlo Simulation', 'ML Model Predictions']
times = [avg_time_mc, avg_time_ml]
colors = ['blue', 'orange']

plt.bar(labels, times, color=colors)
plt.xlabel('Method')
plt.ylabel('Average Time (seconds)')
plt.title('Average Time Taken for Monte Carlo Simulation and ML Model Predictions')
plt.text(0, avg_time_mc + 0.5, f'No. of Records: {num_records}', ha='center', va='bottom')
plt.text(1, avg_time_ml + 0.5, f'No. of Records: {num_records}', ha='center', va='bottom')
plt.show()
