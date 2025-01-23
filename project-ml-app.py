import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("EV_Population.csv")

# Clean and preprocess the data
if 'State' in data.columns:
    data = data.drop(columns=['State'])
if 'Base MSRP' in data.columns:
    data = data.drop(columns=['Base MSRP'])

# App Title
st.write("""
# 🚗 EV Population Analysis App
Analyze and predict trends in electric vehicle populations with ease!  
**Provide input values in the sidebar to make predictions.**
""")

st.markdown("---")

# Sidebar Header
st.sidebar.header("🔧 User Input Parameters")

# Display dataset preview and stats
st.subheader("📊 Dataset Preview")
st.write(data.head())

# Display dataset summary
with st.expander("Dataset Summary"):
    st.write(data.describe())

# Encode categorical columns using LabelEncoder
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split the data into features and target
X = data.drop('Electric Range', axis=1)  # Features
Y = data['Electric Range']  # Target

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# User Input Function
def user_input_features():
    st.sidebar.write("### Input Feature Values")
    inputs = {}

    # Dropdown for 'Make'
    if 'Make' in data.columns:
        make_options = label_encoders['Make'].inverse_transform(range(len(label_encoders['Make'].classes_)))
        selected_make = st.sidebar.selectbox('Select Make', make_options)
        inputs['Make'] = selected_make

    # Dropdown for 'Electric Vehicle Type'
    if 'Electric Vehicle Type' in data.columns:
        type_options = label_encoders['Electric Vehicle Type'].inverse_transform(range(len(label_encoders['Electric Vehicle Type'].classes_)))
        selected_type = st.sidebar.selectbox('Select Electric Vehicle Type', type_options)
        inputs['Electric Vehicle Type'] = selected_type

    # Numeric Features with Sliders
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if column not in ['Electric Range', 'Make', 'Electric Vehicle Type']:  # Exclude non-numeric fields
            min_val = float(data[column].min())
            max_val = float(data[column].max())
            mean_val = float(data[column].mean())

            inputs[column] = st.sidebar.slider(f'{column}', min_val, max_val, mean_val)

    # Convert inputs to DataFrame
    features = pd.DataFrame(inputs, index=[0])

    # Encode categorical inputs
    if 'Make' in features.columns:
        features['Make'] = label_encoders['Make'].transform(features['Make'])
    if 'Electric Vehicle Type' in features.columns:
        features['Electric Vehicle Type'] = label_encoders['Electric Vehicle Type'].transform(features['Electric Vehicle Type'])

    return features

# Get User Input
df = user_input_features()

# Display user input
st.subheader("📝 User Input Parameters")
st.write(df)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Reindex user input to match model training
df = df.reindex(columns=X_train.columns, fill_value=0)

# Predict and Display
try:
    prediction = model.predict(df)
    st.subheader("🔮 Prediction for User Input")
    st.success(f"The predicted Electric Range for the provided input is: **{prediction[0]:.2f} km**")
except ValueError as e:
    st.error(f"Error in prediction: {e}")
    st.warning("Please check your inputs and ensure they match the dataset values.")

# Add footer
st.markdown("---")
st.write("🚀 Developed with ❤️ using Streamlit.")
