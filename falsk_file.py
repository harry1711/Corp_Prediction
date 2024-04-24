from flask import Flask, request, jsonify, render_template,Blueprint
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
site = Blueprint('Site', __name__, template_folder='templates')
# Load the trained model
with open('models\Random_model.pkl', 'rb') as f:
    model = pickle.load(f)

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

    # Preprocess the categorical variables
    input_data_cat_col = pd.DataFrame({
        'District_Name': [district_name],
        'Soil_color': [soil_color],
        'Season': [season]
    })
    input_data_cat_col[['District_Name', 'Soil_color', 'Season']] = input_data_cat_col[['District_Name', 'Soil_color', 'Season']].apply(lambda x: pd.factorize(x)[0])
    # Assuming df is your DataFrame
    #input_data_cat_col[['District_Name', 'Soil_color', 'Season']] = input_data_cat_col[['District_Name', 'Soil_color', 'Season']].applymap(lambda x: abs(x))
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&inputdata_cal_val")
    
    print(input_data_cat_col)

    
# Apply abs() function to specific columns in the DataFrame



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
    print(input_data_combined)

    # Make predictions
    prediction = model.predict(input_data_combined)

    # Return the prediction
    return jsonify(prediction=prediction.tolist())


    # Make predictions
    # prediction = model.predict(input_data_combined)
    # prediction_list = prediction.tolist()
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(prediction_list)

    # prediction_dict = {i: prediction[i] for i in range(len(prediction))}
    # print("predition dict %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(prediction_dict)
    # return jsonify(prediction=prediction_list[0])
    # # Return the prediction
    # return render_template('result_show.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
