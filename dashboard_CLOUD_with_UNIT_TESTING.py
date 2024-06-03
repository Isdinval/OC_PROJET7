# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt

import requests
from io import StringIO

     
        
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
