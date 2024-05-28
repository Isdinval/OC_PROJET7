# -*- coding: utf-8 -*-

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
import pandas as pd
import streamlit as st
import numpy as np
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


st.write('IM HERE')

def preprocess_dataframe(df):
    """
    Preprocesses the input DataFrame (df) for machine learning tasks.
    
    Args:
        df (pandas.DataFrame): The DataFrame to preprocess.
    
    Returns:
        tuple: A tuple containing the following elements:
            - X_train (pandas.DataFrame): The preprocessed training features.
            - y_train (pandas.Series): The training target variable.
            - X_test (pandas.DataFrame): The preprocessed testing features.
            - y_test (pandas.Series): The testing target variable.
    """
    
    # Check for missing values in numerical and categorical features separately
    numerical_missing = df.select_dtypes(include=['int64', 'float64']).isnull().sum().any()
    categorical_missing = df.select_dtypes(include=['object']).isnull().sum().any()
    
    # Define imputation strategies based on missing value presence
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', categorical_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, df.select_dtypes(include=['int64', 'float64']).columns),
        ('categorical', categorical_transformer, df.select_dtypes(include=['object']).columns)
    ])
    
    # Apply preprocessing to the DataFrame
    df_transformed = preprocessor.fit_transform(df)
    

    # Convert the NumPy array back to a DataFrame with column names (without prefixes)
    new_feature_names = []
    for name in preprocessor.get_feature_names_out():
        new_feature_names.append(name.split('__')[1])  # Remove prefix using split and indexing


    # Convert the NumPy array back to a DataFrame with column names
    df_transformed = pd.DataFrame(df_transformed, columns=new_feature_names)

    # Drop preprocessed SK_ID_CURR and add original SK_ID_CURR
    df_transformed.drop('SK_ID_CURR', axis=1, inplace=True)  # Drop preprocessed feature
    df_transformed['SK_ID_CURR'] = df['SK_ID_CURR']  # Add original feature
    
    # Reorder features to place SK_ID_CURR as the first one (efficient way)
    reordered_cols = ['SK_ID_CURR'] + [col for col in df_transformed.columns if col != 'SK_ID_CURR']
    df_transformed = df_transformed[reordered_cols]


    
    return df_transformed
     
     


def get_mlflow_artifact(run_id, artifact_path):
    """Fetches an artifact from the specified MLflow run."""
    client = mlflow.tracking.MlflowClient()
    try:
        local_path = client.download_artifacts(run_id, artifact_path)
        return local_path
    except Exception as e:
        st.error(f"Failed to download artifact: {artifact_path}. Error: {e}")
        return None
    
    
def shap_values_to_dataframe_instance(shap_values, feature_names, instance_index):
    """
    Convert SHAP values to a dataframe indicating the feature and its percentage contribution for a specific instance.
    
    Parameters:
    - shap_values: SHAP values object.
    - feature_names: List of feature names.
    - instance_index: Index of the instance for which to calculate the SHAP values.
    
    Returns:
    - df_feature_importance: Dataframe with features and their percentage contributions for the specific instance.
    """
    # Get SHAP values for the specific instance
    instance_shap_values = shap_values[instance_index]
    
    # Create a dataframe with feature names and their SHAP values for the instance
    df_feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAPValue': instance_shap_values
    })
    
    # # Calculate the percentage contribution for each feature
    # total_shap_value = np.sum(instance_shap_values)
    # df_feature_importance['Percentage'] = 100 * df_feature_importance['SHAPValue'] / total_shap_value
    
    # Sort the dataframe by percentage contribution in descending order
    df_feature_importance = df_feature_importance.sort_values(by='SHAPValue', ascending=False).reset_index(drop=True)
    
    return df_feature_importance


# Define functions to load model and retrieve SHAP values
# @st.cache_resource()  # Cache the model to avoid reloading
def load_model():
  model_name = "XGBOOST_TUNED_FOR_STREAMLIT_CLOUD"
  model_version = 2  # Specify the desired version

  # Load model using model name and version
  model_uri = f"models:/{model_name}/{model_version}"
  try:
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return model
  except MlflowException as e:
    print(f"Error loading model: {e}")
    return None



# @st.cache_data()  # Cache the feature names to avoid reloading
def retrieve_feature_names():
    url = 'https://raw.githubusercontent.com/Isdinval/OC_PROJET7/main/feature_names.txt'
    response = requests.get(url)
    if response.status_code == 200:
        # Read the text content of the response
        text_data = response.text
    
        # Split the text by line breaks to get individual feature names
        feature_names = text_data.splitlines()
        return feature_names
    else:
        st.error("Failed to load data from GitHub.")
        return None




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



# Load the model and SHAP values
model = load_model()
feature_names_from_Model = retrieve_feature_names()
feature_names = feature_names_from_Model

st.write(feature_names)
customer_data = load_test_data()


# Optimal threshold from MLflow
optimal_threshold = 0.636364


  
  
# Define a function to make predictions
def make_prediction(input_data, model, optimal_threshold):
    input_df = pd.DataFrame([input_data])
    # Get the raw prediction score
    probability_class1 = model.predict_proba(input_df)[:, 1]
    # Extract scalar value if probability is an array
    if isinstance(probability_class1, np.ndarray):
        probability_class1 = probability_class1[0]
    # Convert probability to human-readable format
    prediction_label = "Accepted" if probability_class1 >= optimal_threshold else "Refused"
    return probability_class1, prediction_label
  
def get_final_estimator(pipeline):
  """
  Extracts the final estimator from a scikit-learn pipeline.

  Args:
      pipeline: The scikit-learn pipeline object.

  Returns:
      The final estimator object from the pipeline.
  """
  # Access the steps in the pipeline
  steps = pipeline.steps

  # Assuming the final step is the estimator (common case)
  final_estimator_name = steps[-1][0]  # Get the name of the final step
  final_estimator = steps[-1][1]       # Get the final estimator object

  return final_estimator #final_estimator
  

st.write("customer_data")
st.write(customer_data)  
    
# Streamlit app code
def main():
    # Set page title
    st.title("Credit Scoring Dashboard")
    st.write("Welcome to the Credit Scoring Dashboard! Use the form below to make predictions.")
    
    # =========================================================================
    # EXPLAINABILITY SECTIONS
    # =========================================================================
    explainability_sections = """
    MODEL EXPLANATION:
    --    
    This loan approval prediction model is an XGBoost classifier. XGBoost stands for eXtreme Gradient Boosting, a powerful machine learning algorithm that combines the strengths of multiple decision trees to make more accurate predictions. It's known for its efficiency, scalability, and ability to handle complex relationships between features.
    The model analyzes various customer attributes, such as income, credit history, and debt-to-income ratio, to estimate the probability of loan default. The model's output is a probability score between 0 and 1, where a higher score indicates a greater risk of the borrower defaulting on the loan.
    """
    st.write(explainability_sections)



    # Input field for SK_ID_CURR
    sk_id_curr = st.number_input('Enter SK_ID_CURR (ex: 100001 or 101268):', min_value=customer_data['SK_ID_CURR'].min(), max_value=customer_data['SK_ID_CURR'].max())

    # Preprocess Data
    customer_data_preprocessed = preprocess_dataframe(customer_data)

    

    # =========================================================================
    # ADD Missing features Manually (due to preprocess) then re-order features
    # =========================================================================

    # Feature names with spaces (use double quotes)
    new_features = {
        "NAME_FAMILY_STATUS_Unknown": 0,  # Double quotes for space
        "NAME_INCOME_TYPE_Maternity leave": 0,  # Double quotes for space
        "CODE_GENDER_XNA": 0
        }
    # Update customer_data_preprocessed
    customer_data_preprocessed = customer_data_preprocessed.assign(**new_features)

    # Reorder features using "feature_names_from_Model"
    ordered_features = [col for col in feature_names_from_Model if col in customer_data_preprocessed.columns]
    customer_data_preprocessed = customer_data_preprocessed[ordered_features]
    

    # Check if SK_ID_CURR exists in the data
    if sk_id_curr in customer_data_preprocessed['SK_ID_CURR'].values:
        # =========================================================================
        # CUSTOMERS INFORMATIONS
        # =========================================================================
        # Get the index of the selected customer
        customer_index = customer_data_preprocessed[customer_data_preprocessed['SK_ID_CURR'] == sk_id_curr].index[0]

        # Get the data for the selected customer
        input_data = customer_data_preprocessed[customer_data_preprocessed['SK_ID_CURR'] == sk_id_curr].iloc[0].to_dict()




        # =========================================================================
        # PREDICTION USING MODEL FOR SELECTED CUSTOMER
        # =========================================================================
        input_df = pd.DataFrame([input_data])
        probability_class1 = model.predict_proba(input_df)[:, 1]         # Get the raw prediction score

        # Extract scalar value if probability is an array
        if isinstance(probability_class1, np.ndarray):
            probability_class1 = probability_class1[0]
        # Convert probability to human-readable format
        prediction_label = "Accepted" if probability_class1 >= optimal_threshold else "Refused"
        # Display prediction result and probability
        st.markdown("---")
        st.write(f"The probability of non-default on the loan is estimated to be {probability_class1 * 100:.2f}% (Threshold: {optimal_threshold * 100:.2f}%).")

        if prediction_label == "Accepted":
            st.markdown("<p style='text-align: center; font-size: 40px; color: green;'>The loan application is approved.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; font-size: 40px; color: red;'>The loan application is declined.</p>", unsafe_allow_html=True)

        



           

        if prediction_label == "Refused":  
            # =========================================================================
            # SHAP VALUES FOR SELECTED CUSTOMER
            # =========================================================================
            # SHAP VALUES
            final_estimator = get_final_estimator(model)
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(input_df)

            instance_index = 0 # It is always 0 since there is only customer!
            df_feature_importance_instance = shap_values_to_dataframe_instance(shap_values, feature_names, instance_index)
    
            # =========================================================================
            # TOP 10 POSITIVE OR NEGATIVE FEATURES
            # TABLE and MODIFICATION OF VALUES
            # =========================================================================
            # Create two columns for the plots
            col1, col2 = st.columns(2)
    
            # Top 10 Positive Features
            top_10_positive = df_feature_importance_instance.head(10)  # Get the top 10 rows (positive SHAP values)
            
            modified_input_data = input_data.copy()  # Create a copy of the input data
            
            # Plot for Top 10 Positive Features
            with col1:
                st.header("TOP 10 POSITIVE Features")
                st.write(top_10_positive)
                
                
                st.markdown("---")
                st.header("Modify Top 10 Positive Features")
                for feature in top_10_positive['Feature']:
                    try:
                        modified_value = st.number_input(f"POSITIVE - Modify {feature} - Default Value: {round(float(input_data[feature]), 2)}:", value=float(input_data[feature]))
                        modified_input_data[feature] = modified_value
                    except KeyError:
                        # st.error(f"Feature {feature} not found in the input data. Please check the feature names.")
                        pass
            
            # Top 10 Negative Features
            top_10_negative = df_feature_importance_instance.tail(10)  # Get the last 10 rows (negative SHAP values)
            
            # Plot for Top 10 Negative Features
            with col2:
                st.header("TOP 10 NEGATIVE Features")
                st.write(top_10_negative)
                
                st.markdown("---")
                st.header("Modify Top 10 Negative Features")
                for feature in top_10_negative['Feature']:
                    try:
                        modified_value = st.number_input(f"NEGATIVE - Modify {feature} - Default Value: {round(float(input_data[feature]), 2)}:", value=float(input_data[feature]))
                        modified_input_data[feature] = modified_value
                    except KeyError:
                        # st.error(f"Feature {feature} not found in the input data. Please check the feature names.")
                        pass
            


            # =========================================================================
            # UPDATE PREDICTION WITH THE MODIFIED VALUES
            # =========================================================================    
            # Button to make a new prediction with modified data
            if st.button("Re-Predict with Modified Features"):
                probability_class1, prediction_label = make_prediction(modified_input_data, model, optimal_threshold)
                st.write(f"With the modified features, the probability of default on the loan is now estimated to be {probability_class1 * 100:.2f}% (Threshold: {optimal_threshold * 100:.2f}%).")
    
                if prediction_label == "Accepted":
                    st.markdown("<p style='text-align: center; font-size: 40px; color: green;'>The loan application is approved.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='text-align: center; font-size: 40px; color: red;'>The loan application is still declined.</p>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.write("SK_ID_CURR not found.")

if __name__ == "__main__":
    main()
