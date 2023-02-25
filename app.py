# importing Flask
import numpy as np
from flask import Flask, request, render_template
import pickle

# instantiating the app
app = Flask(__name__)

# loading the model from the model.pkl file
model = pickle.load(open("models/model.pkl", "rb"))


# this is the function that the app will run
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template(
        "index.html", prediction_text="Glass Type: {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
