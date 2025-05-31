from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

lr = joblib.load('modelo_regresion_lineal.pkl')
rf = joblib.load('modelo_random_forest.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    print(lr.feature_names_in_)

    data = request.json
    input_df = pd.DataFrame([data])

    input_df = input_df.rename(columns={'Year': 'Año de Asignación'})

    model_features = ['Año de Asignación','Hogares', 'PIB']
    input_df = input_df[model_features]

    lr_pred = lr.predict(input_df)[0]
    rf_pred = rf.predict(input_df)[0]
    ensemble_pred = (lr_pred + rf_pred) / 2

    return jsonify({'prediccion_valor_asignado': ensemble_pred})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
