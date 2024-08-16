from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('SVMmodel.pkl')

# Define the pass thresholds for each hobby
hobby_pass_thresholds = {
    'Painting': 5,
    'Photography': 5,
    'Reading': 7,
    'Playing video games': 7,
    'Driving': 7,
    'Sewing or knitting': 7
}

@app.route('/')
def index():
    return render_template_string("""
           <!doctype html>
           <title>Flask App - SVM</title>
           <h1>Flask App is running</h1>
           <p>To use the predict endpoint, send a POST request to <code>/predict</code> with your raw data.</p>
           """)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print('Received data:', data)  # Debugging line

    # Extracting the necessary fields from the request
    age = data.get('age')
    eyeglasses = data.get('eyeglasses')
    hobby = data.get('hobby')  # Single selected hobby
    preferred_distance = data.get('preferred_distance')
    left_eye_score = data.get('left_eye_score')
    right_eye_score = data.get('right_eye_score')

    # Prepare the input data for left eye
    left_eye_input_data = pd.DataFrame({
        'Age': [age],
        'Eyeglasses': [eyeglasses],
        'Hobby': [hobby],
        'Preferred Distance': [preferred_distance],
        'Score': [left_eye_score]
    })

    # Log the prepared DataFrame
    print('Left Eye Input Data:', left_eye_input_data)

    # Prepare the input data for right eye
    right_eye_input_data = pd.DataFrame({
        'Age': [age],
        'Eyeglasses': [eyeglasses],
        'Hobby': [hobby],
        'Preferred Distance': [preferred_distance],
        'Score': [right_eye_score]
    })

    # Log the prepared DataFrame
    print('Right Eye Input Data:', right_eye_input_data)

    # One-hot encode categorical variables
    left_eye_input_data = pd.get_dummies(left_eye_input_data, columns=['Eyeglasses', 'Hobby'])
    right_eye_input_data = pd.get_dummies(right_eye_input_data, columns=['Eyeglasses', 'Hobby'])

    # Log after one-hot encoding
    print('Left Eye Input Data after one-hot encoding:', left_eye_input_data)
    print('Right Eye Input Data after one-hot encoding:', right_eye_input_data)

    # Ensure all expected columns are present
    expected_columns = model.feature_names_in_
    left_eye_input_data = left_eye_input_data.reindex(columns=expected_columns, fill_value=0)
    right_eye_input_data = right_eye_input_data.reindex(columns=expected_columns, fill_value=0)

    # Log after reindexing
    print('Left Eye Input Data after reindexing:', left_eye_input_data)
    print('Right Eye Input Data after reindexing:', right_eye_input_data)

    # Make predictions
    left_eye_prediction = model.predict(left_eye_input_data)
    right_eye_prediction = model.predict(right_eye_input_data)

    
    print('Left Eye Prediction:', left_eye_prediction)
    print('Right Eye Prediction:', right_eye_prediction)

    pass_threshold = hobby_pass_thresholds.get(hobby, 7)  # Default to 7 if hobby not found

    left_eye_pass = left_eye_score >= pass_threshold
    right_eye_pass = right_eye_score >= pass_threshold

    # Return the prediction results
    return jsonify({
        'left_eye_prediction': 1 if left_eye_pass else 0,
        'right_eye_prediction': 1 if right_eye_pass else 0
    })

if __name__ == '__main__':
    app.run()
