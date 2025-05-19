from flask import Flask, request, jsonify
from pymongo import MongoClient
import joblib
import pandas as pd

app = Flask(__name__)

# —————— Configuración MongoDB ——————
mongo_uri = "mongodb+srv://dvargas:Danik616@bigdata.fy20qda.mongodb.net/?retryWrites=true&w=majority&appName=bigdata"
client = MongoClient(mongo_uri)
db = client.subsidiosDB

# —————— Carga tu modelo ——————
model = joblib.load('modelo_subsidios_final.pkl')

# —————— Endpoint de predicción ——————
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required = ['Departamento','Municipio','Programa','Año de Asignación','Hogares']
    if not all(k in data for k in required):
        return jsonify({'error': f'Faltan campos. Se requieren: {required}'}), 400

    df_input = pd.DataFrame([{
        'Departamento':       data['Departamento'],
        'Municipio':          data['Municipio'],
        'Programa':           data['Programa'],
        'Año de Asignación':  data['Año de Asignación'],
        'Hogares':            data['Hogares']
    }])

    try:
        pred = model.predict(df_input)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'predicted_value': float(pred)})

# —————— Endpoint: lista de programas ——————
@app.route('/programs', methods=['GET'])
def get_programs():
    # extrae todos los documentos de la colección 'programs'
    progs = db.programs.find({}, {'_id': 0, 'Programa': 1})
    # devuelve solo la lista de strings
    programs = [p['Programa'] for p in progs]
    return jsonify({'programs': programs})

# —————— Endpoint: lista de departamentos ——————
@app.route('/departments', methods=['GET'])
def get_departments():
    # Lee de departments_municipalities y extrae solo los nombres de departamento
    docs = db.departments_municipalities.find({}, {'_id': 0, 'Departamento': 1})
    departments = [doc['Departamento'] for doc in docs]
    return jsonify({'departments': departments})

# —————— Endpoint: municipios por departamento ——————
@app.route('/municipalities', methods=['GET'])
def get_municipalities():
    # esperamos el parámetro ?department=ANTIOQUIA
    dept = request.args.get('department')
    if not dept:
        return jsonify({'error': "Falta el parámetro 'department'"}), 400

    # leemos de la colección 'departments_municipalities'
    doc = db.departments_municipalities.find_one(
        {'Departamento': dept},
        {'_id': 0, 'Municipalidades': 1}
    )
    if not doc:
        return jsonify({'error': f"No se encontró departamento '{dept}'"}), 404

    return jsonify({'municipalities': doc['Municipalidades']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
