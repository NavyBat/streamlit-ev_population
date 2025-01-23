import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("EV_Population.csv")

# Remove the 'State' column from the dataset
if 'State' in data.columns:
    data = data.drop(columns=['State'])

# Remove the 'Base MSRP' column from the dataset
if 'Base MSRP' in data.columns:
    data = data.drop(columns=['Base MSRP'])

st.write("""
# EV Population Analysis App
This app analyzes and predicts *Electric Vehicle Population Trends*!
""")

st.sidebar.header('User Input Parameters')

# Display dataset information
st.subheader('Dataset Preview')
st.write(data.head())

# Encode categorical columns using LabelEncoder (once for the entire dataset)
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the encoder for later use

# Split the data into features and target
X = data.drop('Electric Range', axis=1)  # Features
Y = data['Electric Range']  # Target column

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define user input features
def user_input_features():
    inputs = {}

    # Dropdown for 'Make' using original names
    if 'Make' in data.columns:
        make_options = label_encoders['Make'].inverse_transform(range(len(label_encoders['Make'].classes_)))
        selected_make = st.sidebar.selectbox('Select Make', make_options)
        inputs['Make'] = selected_make

    # Dropdown for 'Electric Vehicle Type' using original names (no slider)
    if 'Electric Vehicle Type' in data.columns:
        type_options = label_encoders['Electric Vehicle Type'].inverse_transform(range(len(label_encoders['Electric Vehicle Type'].classes_)))
        selected_type = st.sidebar.selectbox('Select Electric Vehicle Type', type_options)
        inputs['Electric Vehicle Type'] = selected_type

    # Include numeric features with sliders, excluding 'Make' from sliders
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if column != 'Electric Range' and column != 'Make' and column != 'Electric Vehicle Type':  # Excluding target, 'Make', and 'Electric Vehicle Type'
            min_val = float(data[column].min())
            max_val = float(data[column].max())

            # Skip creating slider if min == max
            if min_val == max_val:
                continue  # Skip this column entirely without warning

            mean_val = float(data[column].mean())
            if data[column].dtype in ['int64', 'int32']:
                min_val = int(min_val)
                max_val = int(max_val)
                mean_val = int(mean_val)
                inputs[column] = st.sidebar.slider(f'{column}', min_val, max_val, mean_val)
            else:
                inputs[column] = st.sidebar.slider(f'{column}', min_val, max_val, mean_val)

    # Convert user inputs into a DataFrame
    features = pd.DataFrame(inputs, index=[0])

    # Encode the dropdown selections to match model training
    if 'Make' in features.columns:
        try:
            features['Make'] = label_encoders['Make'].transform(features['Make'])
        except ValueError:
            features['Make'] = label_encoders['Make'].transform([label_encoders['Make'].classes_[0]])  # Use the default label without warning

    if 'Electric Vehicle Type' in features.columns:
        try:
            features['Electric Vehicle Type'] = label_encoders['Electric Vehicle Type'].transform(features['Electric Vehicle Type'])
        except ValueError:
            features['Electric Vehicle Type'] = label_encoders['Electric Vehicle Type'].transform([label_encoders['Electric Vehicle Type'].classes_[0]])  # Use the default label without warning

    return features

# Get user input
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Train a RandomForest model as a regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Ensure user input has the same columns
df = df.reindex(columns=X_train.columns, fill_value=0)  # Ensure input data matches training data columns

# Make predictions for user input
try:
    prediction = model.predict(df)
    st.subheader('Prediction for User Input')
    st.write(f"The predicted Electric Range for the provided input is: *{prediction[0]:.2f} km*")

    # Add graphs after prediction
    st.subheader("Graphs and Insights")

    # Feature Importance Bar Chart
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.write("### Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # Distribution of Electric Range
    st.write("### Distribution of Electric Range")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['Electric Range'], kde=True, color='blue', ax=ax)
    ax.set_title('Distribution of Electric Range')
    st.pyplot(fig)

    # Scatter Plot: Electric Range vs. User's Selected Feature (e.g., Model Year)
    if 'Model Year' in data.columns:
        st.write("### Electric Range vs. Model Year")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=data['Model Year'], y=data['Electric Range'], alpha=0.7, ax=ax)
        ax.set_title('Electric Range vs. Model Year')
        ax.set_xlabel('Model Year')
        ax.set_ylabel('Electric Range (km)')
        st.pyplot(fig)

except ValueError as e:
    st.error(f"Error in prediction: {e}")
    st.warning("Please check your inputs and ensure they match the dataset values.")