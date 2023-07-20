import pytest
import requests
import json
import sys
import pandas as pd
import numpy as np
from flask import jsonify
import re

input_filename = 'input.csv'
input_df = pd.read_csv(input_filename, sep='|')
input_df.replace(np.nan, '', inplace=True)
param_list = (list(input_df.itertuples(index=False, name=None)))


def equal_dict_lists(dict_list_1, dict_list_2, ignore_keys):
    is_equal = False
    for dict_1, dict_2 in zip(dict_list_1, dict_list_2):
        is_equal = equal_dicts(dict_1, dict_2, ignore_keys)
    return is_equal


def equal_dicts(dict_1, dict_2, ignore_keys):
    equal = False
    d1_filtered = dict((k, v) for k, v in dict_1.items() if k not in ignore_keys)
    d2_filtered = dict((k, v) for k, v in dict_2.items() if k not in ignore_keys)
    if d1_filtered == d2_filtered:
        equal = True
    else:
        equal = False
    return equal


def read_json_file_to_dict(json_file):
    f = open(json_file, "r")
    json_dict = json.loads(f.read())
    f.close()
    return json_dict


@pytest.fixture
def api_response():
    def _api_response(method, url, req_json):
        if req_json != '':
            f = open(req_json, "r")
            req_obj = json.loads(f.read())
            f.close()
        else:
            req_obj = ''
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            header_content = {'content-type': 'application/json'}
            response = requests.post(url, data=json.dumps(req_obj), headers=header_content)
        else:
            response = jsonify({'message': 'Invalid request type'})
        return response
    return _api_response


@pytest.mark.parametrize('request_type,url,request_json,response_json,ignore_fields', param_list)
def test_api_call(api_response, request_type, url, request_json, response_json, ignore_fields):
    response = api_response(request_type, url, request_json)
    print('ignore_fields: ', ignore_fields)
    if url.split('/')[-1] == '':
        assert response.status_code == 200
        assert str(response.content.decode('utf-8')) == 'App is running'

    elif url.split('/')[-1] == 'train':
        assert response.status_code == 200
        assert response.json() == {'message': 'Model trained and saved successfully.'}

    elif url.split('/')[-1] == 'predict':
        assert response.status_code == 200
        assert re.match(r"^\d*[.]?\d*$", response.json()['option_value'])

    elif url.split('/')[-1] == 'compareMC':
        assert response.status_code == 200
        benchmark_json_dict = read_json_file_to_dict('responseJSON/compareMC_response.json')
        assert equal_dict_lists(response.json(), benchmark_json_dict['records'], ignore_fields)
        assert re.match(r"^\d*[.]?\d*$", str(response.json()[0]['Option_Value']))
        assert re.match(r"^\d*[.]?\d*$", str(response.json()[1]['Option_Value']))

    elif url.split('/')[-1] == 'read1krecords':
        assert response.status_code == 200
        assert equal_dict_lists(response.json(), read_json_file_to_dict('responseJSON/read1krecords_response.json'), [])

    elif url.split('/')[-1] == 'calcMC':
        assert response.status_code == 200
        assert re.match(r"^\d*[.]?\d*$", str(response.json()['Monte Carlos Option value ']))

    elif url.split('/')[-1] == 'calcGradients':
        assert response.status_code == 200
        assert equal_dicts(response.json(), read_json_file_to_dict('responseJSON/calcGradients_response.json'), [])
