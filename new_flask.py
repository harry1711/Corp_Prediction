from flask import Flask, request, jsonify, render_template, Blueprint
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectFromModel



app = Flask(__name__)
site = Blueprint('Site', __name__, template_folder='templates')

# Load the LabelEncoder object
with open('models/LabelEncoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the trained model
with open('models/Random_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Load the SelectFromModel object
with open('models/Selector.pkl', 'rb') as f:
    selector = pickle.load(f)

# Render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    district_name = request.form.get('District_Name')
    soil_color = request.form.get('Soil_color')
    season = request.form.get('Season')
    nitrogen = float(request.form.get('Nitrogen'))
    phosphorus = float(request.form.get('Phosphorus'))
    potassium = float(request.form.get('Potassium'))
    ph = float(request.form.get('pH'))
    rainfall = float(request.form.get('Rainfall'))
    temperature = float(request.form.get('Temperature'))

    # Load the LabelEncoder object
    with open('models/LabelEncoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Update the LabelEncoder with the new label
    label_encoder['District_Name'].fit([district_name])
    label_encoder['Soil_color'].fit([soil_color])
    label_encoder['Season'].fit([season])

    # Preprocess the categorical variables using the updated LabelEncoder
    district_name_encoded = label_encoder['District_Name'].transform([district_name])[0]
    soil_color_encoded = label_encoder['Soil_color'].transform([soil_color])[0]
    season_encoded = label_encoder['Season'].transform([season])[0]

    # Create a DataFrame with the preprocessed categorical variables
    input_data_cat_col = pd.DataFrame({
        'District_Name': [district_name_encoded],
        'Soil_color': [soil_color_encoded],
        'Season': [season_encoded]
    })

    # Create a DataFrame with numerical features
    input_data_numerical = pd.DataFrame({
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'pH': [ph],
        'Rainfall': [rainfall],
        'Temperature': [temperature]
    })

    # Concatenate the categorical and numerical data
    input_data_combined = pd.concat([input_data_cat_col, input_data_numerical], axis=1)


    # Perform feature selection
    selected_features = selector.transform(input_data_combined)

    #det label
    # Make predictions using the selected features
    prediction_labels = model.predict(selected_features)

    # Make predictions using the selected features
    prediction_probs = model.predict_proba(selected_features)[0]

    # Get the indices of the top 3 probabilities
    top_3_indices = np.argsort(prediction_probs)[::-1][:3]

    # Get the top 3 crop names
    top_3_crops = [label_encoder['Crop'].inverse_transform([index])[0] for index in top_3_indices]

    top_3_percentages = [round(prediction_probs[index] * 100, 2) for index in top_3_indices]

    # Create a list of dictionaries containing crop names and percentages
    prediction_list = [{'crop': crop, 'percentage': percentage} for crop, percentage in zip(top_3_crops, top_3_percentages)]

    # Return the top 3 crop names and their percentages
    return jsonify(prediction=prediction_list)

    # Make predictions using the selected features
    #prediction = model.predict(selected_features)
    # Make predictions
    #prediction = model.predict(input_data_combined)
    # Return the prediction
    #return jsonify(prediction=prediction.tolist())


if __name__ == '__main__':
    app.run(debug=True)