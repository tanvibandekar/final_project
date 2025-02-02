from flask import Flask, jsonify, request
from flask import Flask, render_template

from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load the trained model and scaler
model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Function to calculate CL and Vd from patient demographics
def calculate_cl_vd(weight, height):
    bmi = weight / (height / 100) ** 2
    bsa = np.sqrt((height * weight) / 3600)
    CL = 13.2 * (weight / 70.0) ** 0.75  # Example base clearance (L/h)
    Vd = 172 * (weight / 70.0)  # Example base volume (L)
    return CL, Vd, bmi, bsa

# Function to predict concentration for a new patient
def predict_concentration(age, weight, height, time_point):
    CL, Vd, bmi, bsa = calculate_cl_vd(weight, height)
    patient_data_with_cl_vd = [age, weight, height, bmi, bsa, CL, Vd]
    patient_data_scaled = scaler.transform([patient_data_with_cl_vd + [time_point]])
    predicted_concentration = model.predict(patient_data_scaled)
    
    # output = {
    #     "Age": age,
    #     "Weight": weight,
    #     "Height": height,
    #     "Time": time_point,
    #     "Predicted Concentration": round(predicted_concentration[0][0], 4)
    # }
    # return output
    print(predicted_concentration[0][0])
    return float(round(predicted_concentration[0][0], 4))


# Create the Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# # Form route
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
    
#     output = None  # Default value for output
#     if request.method == "POST":
                
#         try:
#             # Get form data
#             age = request.form.get("age", type=float)
#             weight = request.form.get("weight", type=float)
#             height = request.form.get("height", type=float)
#             time = request.form.get("time", type=float)
                    
#             # Perform the calculation for predicted concentration
#             result = None  # Initialize result
#             if age and weight and height and time:

#                 # Predict concentration
#                 prediction = predict_concentration(age, weight, height, time) 

#         except Exception as e:
#             prediction = {"Error": str(e)}
        
#     return render_template("predict.html", prediction=prediction)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    output = {"prediction": None}  # Default response

    if request.method == "POST":
        try:
            # Get form data
            age = request.form.get("age", type=float)
            weight = request.form.get("weight", type=float)
            height = request.form.get("height", type=float)
            time = request.form.get("time", type=float)

            # Ensure all inputs are provided
            if None not in (age, weight, height, time):
                # Predict concentration
                prediction = predict_concentration(age, weight, height, time)

                print("Raw Prediction Output:", prediction)  # Debugging

                # Check if the output is a dictionary (incorrect output)
                if isinstance(prediction, dict):
                    output["error"] = f"Unexpected dictionary output: {prediction}"
                else:
                    output["prediction"] = float(prediction)  # Convert to standard float

            else:
                output["error"] = "Missing input values"

        except Exception as e:
            output["error"] = str(e)

    return jsonify(output)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
