# import libraris
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

## col names for df
col_names = ['Dependents', 
             'ApplicantIncome', 
             'CoapplicantIncome', 
             'LoanAmount',
            'Loan_Amount_Term', 
            'Credit_History', 
            'Gender_Female', 
            'Gender_Male',
            'Married_No', 
            'Married_Yes', 
            'Education_Graduate',
            'Education_Not Graduate', 
            'Self_Employed_No', 
            'Self_Employed_Yes',
            'Property_Area_Rural', 
            'Property_Area_Semiurban', 
            'Property_Area_Urban',
            'Loan_Status_N', 
            'Loan_Status_Y']


# initialzie the flask app
app = Flask(__name__)

# load ml model
# model = pickle.load(open('model.pkl', 'rb'))

# define the app route for the default page of the web-app
@app.route('/')
def home():
    return render_template('index.html')

 # Get the user input from the HTML form
    input_data = []
    for col in col_names:
        input_data.append(float(request.form[col]))

    # Convert the input data into a DataFrame
    df_XX = pd.DataFrame([input_data], columns=col_names)

    # Make the prediction
    prediction = model.predict(df_XX)
    loan_approval_status = 'Approved' if prediction[0] == 1 else 'Not Approved'

    return render_template('result.html', status=loan_approval_status)


# start the flask server
if __name__ == '__main__':
    app.run(debug=True)