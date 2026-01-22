from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/house_price_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        overall_qual = float(request.form["OverallQual"])
        gr_liv_area = float(request.form["GrLivArea"])
        total_bsmt_sf = float(request.form["TotalBsmtSF"])
        garage_cars = float(request.form["GarageCars"])
        bedroom_abv_gr = float(request.form["BedroomAbvGr"])
        full_bath = float(request.form["FullBath"])

        # Arrange input in same order used during training
        features = np.array([[overall_qual, gr_liv_area, total_bsmt_sf,
                              garage_cars, bedroom_abv_gr, full_bath]])

        # Scale input
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ₦{prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error occurred. Please check your inputs."
        )

if __name__ == "__main__":
    app.run(debug=True)
