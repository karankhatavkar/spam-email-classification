from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Add pickel
vectorizer = pickle.load(open("model/vectorizer.pkl", 'rb'))
model = pickle.load(open("model/model.pkl", 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["post"])
def predict():
    email = request.form.get('email')
    
    # Prdict Email 
    prediction = model.predict(vectorizer.transform([email]))
    prediction = 1 if prediction == 1 else -1

    return render_template('index.html', response=prediction)


# if __name__ == "__main__":
#     app.run()