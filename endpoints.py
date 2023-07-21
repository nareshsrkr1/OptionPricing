from flask import Blueprint, jsonify, request, current_app
from sklearn.model_selection import train_test_split
from OptionTrainNBuild import load_dataset, scale_data, build_model, save_model, model_custom_predict, model_custom_predict_multiple,read1krecords
import traceback
from torch.autograd import Variable
from ComputeBS_MC import black_scholes_call,monte_carlo_call
import torch
from log_conf import logger,set_log_filename,initialize_log_handler


model_routes = Blueprint('model_routes', __name__)

@model_routes.route('/')
def home():
    set_log_filename('logs/test.log')
    initialize_log_handler()
    logger.info('App is running')
    print('App is running')
    return 'App is running'

@model_routes.route('/train', methods=['POST'])
def train_model():
    try:
        set_log_filename('logs/model_train.log')
        initialize_log_handler()
        # Load and preprocess the dataset
        dataset_filename = current_app.config['input']['dataset_filename']
        X, Y = load_dataset(dataset_filename)
        logger.info('Input Dataset loaded')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        logger.info('Scale data is complete')

        # Build the model
        model = build_model(X_train_scaled.shape[1])
        logger.info('Model build is complete')


        # Train the model
        num_epochs = 100
        batch_size = 64
        logger.info('Training Started')

        model.fit(X_train_scaled, Y_train, batch_size=batch_size, epochs=num_epochs,
                  validation_split=0.1, verbose=2)
        logger.info('Training complete')

        # Evaluate model on test data
        test_loss = model.evaluate(X_test_scaled, Y_test)
        test_accuracy = 100 - test_loss * 100
        print("Test Accuracy: {:.2f}%".format(test_accuracy))
        logger.info("Test Accuracy: {:.2f}%".format(test_accuracy))


        # Save the model and scaler
        model_filename = current_app.config['files']['model_filename']
        scaler_filename = current_app.config['files']['scaler_filename']
        save_model(model, scaler, model_filename, scaler_filename)
        logger.info('Model and scalars saved')

        return jsonify({'message': 'Model trained and saved successfully.'})
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during model training.'})


@model_routes.route('/predict', methods=['POST'])
def predict_option_value():
    try:
        set_log_filename('logs/model_predict.log')
        initialize_log_handler()
        data = request.json
        spot_price = float(data['Spot_Price'])
        strike_price = float(data['Strike_Price'])
        maturity = float(data['Maturity'])
        risk_free_interest = float(data['risk_free_interest'])
        volatility = float(data['Volatility'])
        model_filename = current_app.config['files']['model_filename']
        scaler_filename = current_app.config['files']['scaler_filename']
        option_value = model_custom_predict(spot_price, strike_price, maturity, risk_free_interest, volatility,model_filename,scaler_filename)
        logger.info("Option value prediction: %s", option_value)
        return jsonify({'option_value': str(round(option_value[0][0],2))})
    except Exception as e:
        # print(str(e))
        logger.error("An error occurred during option value prediction: %s", str(e))
        return jsonify({'error': 'An error occurred during option value prediction.'})

@model_routes.route('/compareMC', methods=['POST'])
def compareMC():
    try:
        set_log_filename('logs/compareMC.log')
        initialize_log_handler()
        json_data = request.json
        loaded_model = current_app.loaded_model
        scaler = current_app.scaler
        calc_option_values_json = model_custom_predict_multiple(json_data,loaded_model,scaler)

        return jsonify(calc_option_values_json)

    except Exception as e:
        logger.error("An error occurred during option value prediction: %s", str(e))
        return jsonify({'error': 'An error occurred during option value prediction. ' + str(e)})

@model_routes.route('/read1krecords', methods=['GET'])
def send1krecords():
    try:
        set_log_filename('logs/read1krecords.log')
        initialize_log_handler()
        file_name = current_app.config['input']['read1k_filename']
        results = read1krecords(file_name)
        logger.info("fetched random1k records file ")
        return jsonify((results))
    except Exception as e:
        logger.error("An error occurred during option value prediction: %s", str(e))
        return jsonify({'error': 'An error occurred during option value prediction. ' + str(e)})

#Endpoint to calculate Monte Carlos
@model_routes.route('/calcMC', methods=['POST'])
def calcMonteCarlos():
    try:
        set_log_filename('logs/calcMC.log')
        initialize_log_handler()
        # Get the input data from the request
        data = request.json
        S = Variable(torch.tensor(float(data['Spot_Price'])), requires_grad=True)
        K = Variable(torch.tensor(float(data['Strike_Price'])), requires_grad=True)
        r = Variable(torch.tensor(float(data['risk_free_interest'])), requires_grad=True)
        T = Variable(torch.tensor(float(data['Maturity'])/365), requires_grad=True)
        sigma = Variable(torch.tensor(float(data['Volatility'])), requires_grad=True)
        call_option_value = monte_carlo_call(S,K,r,T,sigma)
        logger.info("Calculate option value "+ str(call_option_value.item()))
        return jsonify({'Monte Carlos Option value ': round(call_option_value.item(),2)})
    except Exception as e:
        logger.error("An error occurred during option value prediction: %s", str(e))
        return jsonify({'error': 'An error occurred during option value prediction. ' + str(e)})

#Endpoint to calculate Monte Carlos
@model_routes.route('/calcGradients', methods=['POST'])
def calcBSNGradients():
    try:
        set_log_filename('logs/calcGrad.log')
        initialize_log_handler()
        # Get the input data from the request
        data = request.json
        S = Variable(torch.tensor(data['Spot_Price']), requires_grad=True)
        K = Variable(torch.tensor(data['Strike_Price']), requires_grad=True)
        r = Variable(torch.tensor(data['risk_free_interest']), requires_grad=True)
        T = Variable(torch.tensor(data['Maturity']/365), requires_grad=True)
        sigma = Variable(torch.tensor(data['Volatility']), requires_grad=True)

        call_price = black_scholes_call(S, K, r, T, sigma)
        logger.info("Calculate option value "+ str(call_price.item()))

        call_price.backward()

        # Access the gradients
        dS = S.grad
        dK = K.grad
        dr = r.grad
        dT = T.grad
        dsigma = sigma.grad

        return jsonify({
            'Black Scholes calculated Option value': round(call_price.item(), 2),
            'dS': dS.item(),
            'dK': dK.item(),
            'dr': dr.item(),
            'dT': dT.item(),
            'dSigma': dsigma.item()
        })

    except Exception as e:
        logger.error("An error occurred during option value prediction: %s", str(e))
        return jsonify({'error': 'An error occurred during option value prediction. ' + str(e)})
