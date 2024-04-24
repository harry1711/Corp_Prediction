from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def Get_form_data():
    return "Hello world..I am Himmat"

@app.route('/predict', methods=['GET','POST'])
def predict():
    print(model)
    District = request.form.get('District')
    SoilColor = request.form.get('SoilColor')
    Season = request.form.get('Season')
    Nitrogen = request.form.get('Nitrogen')
    Phosphorus = request.form.get('Phosphorus')
    Potassium = request.form.get('Potassium')
    pH = request.form.get('pH')
    Rainfall = request.form.get('Rainfall')
    Temperature = request.form.get('Temperature')
    print("in predict funtion")

    input_query = np.array([[District, SoilColor, Season, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature]])

    results = model.predict(input_query)
    print(results)
    return {'District':District}

    #output = {'Sugarcane': str(results[0]), 'Wheat': str(results[1]), 'Rice': str(results[2])}

   #return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
    
