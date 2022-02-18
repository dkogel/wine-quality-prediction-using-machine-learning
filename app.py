from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

X_red_scaler = pickle.load( open( "X_red_scaler.pkl", "rb" ) )
rf_red = pickle.load( open( "rf_red.pkl", "rb" ) )


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/decision_tree")
def dt():

    return render_template("decision_tree.html")

@app.route("/knn")
def knn():

    return render_template("KNN.html")

@app.route("/LogisticRegression")
def lr():

    return render_template("LogisticRegression.html")

@app.route("/neural_network")
def nn():

    return render_template("neural_network.html")

@app.route("/SVM")
def svm():

    return render_template("SVM.html")

@app.route("/random_forest")
def random_forest():

    return render_template("random_forest.html", result="")


# route to accept information from webpage
@app.route("/form", methods=["POST"])
def form():
    if request.method == "POST":
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugars = float(request.form["residual_sugars"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        dictionary = {"fixed_acidity": [fixed_acidity],
                      "volatile_acidity": [volatile_acidity],
                      "citric_acid": [citric_acid],
                      "residual_sugars": [residual_sugars],
                      "chlorides": [chlorides],
                      "free_sulfur_dioxide": [free_sulfur_dioxide],
                      "total_sulfur_dioxide":[total_sulfur_dioxide],
                      "density": [density],
                      "pH": [pH],
                      "sulphates": [sulphates],
                      "alcohol": [alcohol] }

        df = pd.DataFrame(dictionary, index=None)


        final_features_scaled = X_red_scaler.transform(df)


        prediction = rf_red.predict(final_features_scaled)
        # prediction_labels =["a","b","c","d","e","f"]
        result = prediction[0] 
                    


    return render_template("random_forest.html", result= result, scroll= "scroll_true")

if __name__== "__main__":
    app.debug = True
    app.run(use_reloader=False)

