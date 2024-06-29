import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from BoxingRecognition.util import DynamicTimeWarpingUtility

app = Flask(__name__)
CORS(app)


model, feature_names = joblib.load("../BoxingRecognition/models/decision_tree_model_and_features.pkl")


@app.route("/deprecated", methods=['POST'])
def predict_features():
    # Validate JSON request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Define required sensor fields and statistics
    required_fields = {"accelerometer": ["x", "y", "z"], "gyroscope": ["x", "y", "z"]}
    stats_fields = ["mean", "min", "max"]

    # Initialize a dictionary to hold the structured data for prediction
    structured_data = {}

    # Check for missing sensors and axes
    for sensor, axes in required_fields.items():
        if sensor not in data:
            return jsonify({"error": f"Missing sensor data for {sensor}"}), 400
        for axis in axes:
            if axis not in data[sensor]:
                return jsonify({"error": f"Missing axis {axis} data for {sensor}"}), 400
            for stat in stats_fields:
                key_name = f"{sensor}_{axis}_{stat}"
                if stat not in data[sensor][axis]:
                    return jsonify({"error": f"Missing {stat} data for {sensor} (axis {axis})"}), 400
                structured_data[key_name] = data[sensor][axis][stat]

    # Check if all required features are present and in the correct order
    try:
        feature_vector = [structured_data[fn] for fn in feature_names]
    except KeyError as e:
        return jsonify({"error": f"Missing data for {str(e)}"}), 400

    # Create DataFrame for prediction
    df = pd.DataFrame([feature_vector], columns=feature_names)

    # Make prediction
    prediction = model.predict(df)[0]

    # Return prediction in JSON format
    return jsonify({"prediction": prediction}), 200


# Import the templates
left_slip_template = np.loadtxt('../Data/dtw/left_slip_concatenated_template.csv', delimiter=',')
right_slip_template = np.loadtxt('../Data/dtw/right_slip_concatenated_template.csv', delimiter=',')
left_roll_template = np.loadtxt('../Data/dtw/left_roll_concatenated_template.csv', delimiter=',')
right_roll_template = np.loadtxt('../Data/dtw/right_roll_concatenated_template.csv', delimiter=',')
pull_back_template = np.loadtxt('../Data/dtw/pull_back_concatenated_template.csv', delimiter=',')
templates = {
    'Left Slip': left_slip_template,
    'Right Slip': right_slip_template,
    'Left Roll': left_roll_template,
    'Right Roll': right_roll_template,
    'Pull Back': pull_back_template
}


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    prediction, distance, path = DynamicTimeWarpingUtility.classify_sequence(data, templates)
    return jsonify({"prediction": prediction, "distance": distance, "path": path}), 200

if __name__ == '__main__':
    app.run(debug=True)  # Runs the server in debug mode
