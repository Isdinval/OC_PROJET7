# -*- coding: utf-8 -*-
import pandas as pd
import requests
import unittest
import unittest.mock
import streamlit as st
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
import shap
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

import requests
from io import StringIO


# =========================================================================
# INITIAL FUNCTIONS
# =========================================================================           
@st.cache_resource()  
def load_model():
    # Set the MLflow tracking URI (update with your server URI if necessary)
    mlflow.set_tracking_uri("https://dagshub.com/Isdinval/OC_PROJET7.mlflow")
    # Define the model URI from the provided information
    model_uri = 'runs:/19e1265fed5543db8878f67479e4f60b/model'
    # Load the model using the appropriate method
    model = mlflow.sklearn.load_model(model_uri)
    return model
# Load the test data
@st.cache_data()   # Cache the test data to avoid reloading
def load_test_data():
    url = 'https://raw.githubusercontent.com/Isdinval/OC_PROJET7/main/application_test.csv'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text), delimiter=",")
    else:
        st.error("Failed to load data from GitHub.")
        return None
    return feature_names



# =========================================================================
# UNIT TESTS (USING UNITTEST)
# =========================================================================       
import unittest
class TestHelperFunctions(unittest.TestCase):
    
    @unittest.mock.patch('mlflow.sklearn.load_model')
    def test_load_model(self, mock_load_model):
        # Mock successful model loading
        mock_load_model.return_value = "Mocked Model"
        loaded_model = load_model()
        self.assertEqual(loaded_model, "Mocked Model")  # Assert returned value
      
        # Mock unsuccessful model loading (optional)
        mock_load_model.side_effect = Exception("Mocked Error")
        with self.assertRaises(Exception):
            load_model()

    @unittest.mock.patch('requests.get')
    def test_load_test_data(self, mock_get):
        # Mock successful data loading
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "header1,header2\nvalue1,value2"
        mock_get.return_value = mock_response
      
        df = load_test_data()
        self.assertIsInstance(df, pd.DataFrame)  # Assert returned value is a DataFrame
        self.assertEqual(df.shape, (2, 2))  # Assert data has 2 rows and 2 columns (based on mock data)
      
        # Mock failed data loading (optional)
        mock_get.return_value.status_code = 500
        with self.assertRaises(Exception):  # Expect an exception on failed request
            load_test_data()

if __name__ == "__main__":
    unittest.main()
    main()
