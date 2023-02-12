# Import the necessary libraries
from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the KNN model
knn = pickle.load(open("knn.pkl", "rb"))

# Initialize the Flask app
app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# Define the route for the prediction page
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the input values from the form
        BMI = float(request.form["bmi"])
        Smoking = int(request.form["smoke"])
        AlcoholDrinking = int(request.form["alcohol"])
        Stroke = int(request.form["stroke"])
        PhysicalHealth = float(request.form["physicalH"])
        MentalHealth = float(request.form["mentalH"])
        Sex = int(request.form["sex"])
        AgeCategory = int(request.form["age"])
        Diabetic = int(request.form["diabetic"])
        PhysicalActivity = int(request.form["pActivity"])
        GenHealth = int(request.form["genHealth"])
        SleepTime = float(request.form["sleep"])
        Asthma = int(request.form["asthama"])
        KidneyDisease = int(request.form["kidney"])
        SkinCancer = int(request.form["skin"])
        
        # Create the input vector
        x = np.array([[BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, Sex, AgeCategory, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer]]).reshape(1, -1)
        print(x)        
        # Use the model to make the prediction
        prediction = knn.predict(x)[0]
        
        # Render the prediction page with the prediction
        return render_template("predict.html", prediction=prediction)
    
    # Render the form if the request method is GET
    return render_template("index.html")

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
