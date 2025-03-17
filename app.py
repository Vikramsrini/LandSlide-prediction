from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Your prediction logic here
    prediction = 1  # Example prediction
    probability = 0.85  # Example probability
    return jsonify({
        'prediction': prediction,
        'probability': probability
    })

if __name__ == '__main__':
    app.run(debug=True)