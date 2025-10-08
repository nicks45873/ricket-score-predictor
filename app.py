
from flask import Flask, request, jsonify
import joblib
import pandas as pd

model = joblib.load("cricket_score_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "🏏 Cricket Score Prediction API with Team Names is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    team_name = data.get('team_name', 'Unknown')
    overs = data.get('overs')
    wickets = data.get('wickets')
    run_rate = data.get('run_rate')
    current_score = data.get('current_score')

    input_data = pd.DataFrame([[overs, wickets, run_rate, current_score]],
                              columns=['overs', 'wickets', 'run_rate', 'current_score'])
    predicted_score = model.predict(input_data)[0]

    return jsonify({
        "team_name": team_name,
        "predicted_final_score": round(float(predicted_score), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
