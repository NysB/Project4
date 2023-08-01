# import dependencies
import pandas as pd

import numpy as np
from flask import Flask, render_template, request
import pickle

# col names for df
input_fields = ["Gender",
                "Married",
                "Dependents",
                "Education", 
                "Self_Employed",
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History",
                "Property_Area"]

# initialzie the flask app
app = Flask(__name__)

# load logistic regression model
model = pickle.load(open('logistic_regression.pkl', 'rb'))

# define the app 

@app.route('/')
def home():
    return render_template('index.html')

# to use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():

    ## Input variables

    X = np.zeros(len(input_fields))
    print("X",X)
    predictions_df = pd.DataFrame(data=[dict(zip(input_fields, X))])
   
    gender_input = request.form['Gender']
    print("Gender:", gender_input)
    predictions_df['Gender'] = gender_input

    married_input = request.form['Married']
    print("Marital Status:", married_input)
    predictions_df['Married'] = married_input

    dependents_input = request.form['Dependents']
    print("Dependents:", dependents_input)
    predictions_df['Dependents'] = dependents_input

    education_input = request.form['Education']
    print("Education:", education_input)
    predictions_df['Education'] = education_input

    self_employed_input = request.form['Self_Employed']
    print("Self Employed:", self_employed_input)
    predictions_df['Self_Employed'] = self_employed_input

    applicant_income_input = request.form['ApplicantIncome']
    print("Applicant Income:", applicant_income_input)
    predictions_df['ApplicantIncome'] = applicant_income_input

    co_applicant_income_input = request.form['CoapplicantIncome']
    print("Co-Applicant Income:", co_applicant_income_input)
    predictions_df['CoapplicantIncome'] = co_applicant_income_input

    loan_amount_input = request.form['LoanAmount']
    print("Gender:", loan_amount_input)
    predictions_df['LoanAmount'] = loan_amount_input

    loan_amount_term_input = request.form['Loan_Amount_Term']
    print("Loan Term:", loan_amount_term_input)
    predictions_df['Loan_Amount_Term'] = loan_amount_term_input

    credit_history_input = request.form['Credit_History']
    print("Credit History:", credit_history_input)
    predictions_df['Credit_History'] = credit_history_input

    property_area_input = request.form['Property_Area']
    print("Gender:", property_area_input)
    predictions_df['Property_Area'] = property_area_input

    ## Make prediction

    prediction = model.predict_proba(predictions_df)
    print("prediction",prediction)
    
    output = np.round(prediction[0][1], 2)

    print('You are likely: {}'.format(output))

    if output > (.60):
        page = "reject.html"
    else:
        page = "approve.html"
    return render_template(page, prediction_text='Probability: {}'.format(output))

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)