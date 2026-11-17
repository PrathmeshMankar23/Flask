#app.py files handles loading the trained model

from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np
app = Flask(__name__)

#load the trained model
with open('model (1).pkl', 'rb') as file:
    model = pickle.load(file)
mapping_dict = {'income':{0:'<=50',1:'>50k'}}
@app.route('/')
def home():
    
    return render_template('index.html')
@app.route('/result', methods=['POST'])
def result():
    try:
        features = [
            int(request.form['age']),
            int(request.form['workclass']),
            int(request.form['education-num']),
            int(request.form['marital-status']),
            int(request.form['occupation']),
            int(request.form['relationship']),
            int(request.form['race']),
            int(request.form['gender']),
            int(request.form['capital-gain']),
            int(request.form['capital-loss']),
            int(request.form['hours-per-week']),
            int(request.form['native-country'])
        ]
        prediction = model.predict([features])[0]
        predicted_income = mapping_dict['income'][prediction]
        return render_template('index.html', prediction_text=f"Predicted Income: {predicted_income}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
    
   