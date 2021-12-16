from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder='template')
model = joblib.load('regressor.pkl', 'r')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        pred = np.expm1(model.predict(final_features))
        output_val = pred

        return render_template('predictor.html', prediction_text = 'The car you are looking for should cost: {}'.format(output_val))

    return render_template('predictor.html')

if __name__ == '__main__':
    app.run(debug=True)