import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# 1. FIX PATH ISSUES: Tell Flask exactly where to find your files
# This gets the folder where app.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# This tells Flask the templates folder is right next to app.py
template_dir = os.path.join(base_path, 'templates')

app = Flask(__name__, template_folder=template_dir)

# 2. LOAD MODELS (Using absolute paths to prevent "File Not Found" errors)
try:
    model = joblib.load(os.path.join(base_path, 'stream_model.pkl'))
    gender_encoder = joblib.load(os.path.join(base_path, 'gender_encoder.pkl'))
    stream_encoder = joblib.load(os.path.join(base_path, 'stream_encoder.pkl'))
    print(" Model and Encoders loaded successfully!")
except Exception as e:
    print(f" Error loading models: {e}")

@app.route('/')
def index():
    # This route shows the initial form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get values from the HTML form
            gender = request.form['gender']
            math = int(request.form['math'])
            science = int(request.form['science'])
            arts = int(request.form['arts'])
            hours = int(request.form['hours'])

            # 2. Preprocess the input (convert 'Male'/'Female' to numbers)
            gender_encoded = gender_encoder.transform([gender])[0]
            
            # 3. Create a DataFrame for the model
            input_df = pd.DataFrame([[gender_encoded, math, science, arts, hours]], 
                                    columns=['Gender', 'Math_Marks', 'Science_Marks', 'Arts_Marks', 'Study_Hours_Per_Week'])

            # 4. Make Prediction
            prediction_num = model.predict(input_df)
            result = stream_encoder.inverse_transform(prediction_num)[0]
            
            # 5. Calculate Confidence (Probability)
            prob = model.predict_proba(input_df).max() * 100

            # 6. Return results back to the website
            return render_template('index.html', 
                                   prediction_text=result, 
                                   confidence=f"{prob:.2f}%",
                                   math=math, science=science, arts=arts,
                                   show_results=True)
        
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
     app.run(debug=True)