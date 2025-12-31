import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


st.write("""
# Crop Yield Project

In this project, yield per hectare (tons) is calculated by taking into account the variables 
         Region, Soil_Type, Crop, Weather, Rainfall, Temperature, Days_to_Harvest, Fertilizer_Used, and Irrigation_Used.
         """)

st.sidebar.header("User Input Features")


#----------User's Input Function----------
def user_input_features():
    Region = st.sidebar.selectbox(
        'Region',
        ('North', 'East', 'South', 'West')
    )

    Soil_Type = st.sidebar.selectbox(
        'Soil Type',
        ('Clay', 'Sandy', 'Loam', 'Silt', 'Peaty', 'Chalky')
    )

    Crop = st.sidebar.selectbox(
        'Crop',
        ('Wheat', 'Rice', 'Maize', 'Barley', 'Soybean', 'Cotton')
    )

    Weather = st.sidebar.selectbox(
        'Weather',
        ('Sunny', 'Rainy', 'Cloudy')
    )

    Rainfall = st.sidebar.slider(
        'Rainfall (mm)',
        min_value=100.01422392498176,
        max_value=999.9842154883752,
        value=800.0
    )

    Temperature = st.sidebar.slider(
        'Temperature (°C)',
        min_value=0.0,
        max_value=39.998955324275066,
        value=20.0
    )

    Days_to_Harvest = st.sidebar.slider(
        'Days to Harvest',
        min_value=60,
        max_value=149,
        value=110
    )

    Fertilizer_Used = st.sidebar.selectbox(
        'Fertilizer Used',
        (True, False)
    )

    Irrigation_Used = st.sidebar.selectbox(
        'Irrigation Used',
        (True, False)
    )

    data = {
        'Region': Region,
        'Soil_Type': Soil_Type,
        'Crop': Crop,
        'Weather': Weather,
        'Rainfall': Rainfall,
        'Temperature': Temperature,
        'Days_to_Harvest': Days_to_Harvest,
        'Fertilizer_Used': Fertilizer_Used,
        'Irrigation_Used': Irrigation_Used
    }

    return pd.DataFrame([data])


#----------User's Inputs----------
input_df = user_input_features()



#----------Reading Cleaning Dataset----------
df = pd.read_csv("cleaned_dataset.csv") 



#----------Splitting Dataset to Dependent and Independent Data----------
X = df.drop(columns=["Yield"])
y = df["Yield"]



#----------Splitting Data----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



#----------Numerical, Boolean and Categorical Columns----------
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns
bool_cols = X_train.select_dtypes(include=["bool"]).columns



#----------Scaling the Data----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols), 
        ("bool", "passthrough", bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop='first'), cat_cols)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
user_input_encoded = preprocessor.transform(input_df)



#----------Model Fit and Prediction----------
X_train_sm = sm.add_constant(X_train_encoded)
model = sm.OLS(y_train, X_train_sm).fit()

user_input_encoded_sm = sm.add_constant(user_input_encoded, has_constant='add')

prediction = model.predict(user_input_encoded_sm)



#---------- Main Panel----------
st.subheader("Predict Result")
st.success(f"Predicted Yield: {prediction[0]:.2f} ton")



with st.expander("More"):

    st.subheader("User's Inputs")
    st.write(input_df)


    st.divider()
    st.subheader("User's Inputs Encoded")
    st.write(user_input_encoded_sm)


    st.divider()
    st.subheader("Model Summary")
    st.write(model.summary())


    st.divider()
    st.subheader("Metrics")
    y_test_pred = model.predict(sm.add_constant(X_test_encoded))
    st.write("MAE: ", mean_absolute_error(y_test, y_test_pred))
    st.write("RMSE: ", root_mean_squared_error(y_test, y_test_pred))
    st.write("R2: ", r2_score(y_test, y_test_pred))