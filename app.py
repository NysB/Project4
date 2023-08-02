# import dependencies
import pandas as pd

import numpy as np
from flask import Flask, render_template, request
import pickle

# initialzie the flask app
app = Flask(__name__)

# load logistic regression model
model = pickle.load(open('logistic_regression.pkl', 'rb'))
x_scaler = pickle.load(open('x_scaler.pkl', 'rb'))

# define the app 

@app.route('/')
def home():
    return render_template('index.html')

# to use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():

    ## Input variables
   
    applicant_income_input = float(request.form['ApplicantIncome'])

    co_applicant_income_input = float(request.form['CoapplicantIncome'])

    loan_amount_input = float(request.form['LoanAmount'])

    loan_amount_term_input = float(request.form['Loan_Amount_Term'])

    credit_history_input = float(request.form['Credit_History'])

    gender_input = request.form['Gender']

    if gender_input == "Male":
        gender_male = 1
        gender_female = 0
    else:
        gender_male = 0
        gender_female = 1      

    married_input = request.form['Married']

    if married_input == "Y":
        married_no = 0
        married_yes = 1
    else:
        married_no = 1
        married_yes = 0

    dependents_input = request.form['Dependents']

    if dependents_input == "0":
        dependents_0 = 1
        dependents_1 = 0
        dependents_2 = 0
        dependents_3 = 0
    elif dependents_input == "1":
        dependents_0 = 0
        dependents_1 = 1
        dependents_2 = 0
        dependents_3 = 0
    elif dependents_input == "2":
        dependents_0 = 0
        dependents_1 = 0
        dependents_2 = 1
        dependents_3 = 0
    else:
        dependents_0 = 0
        dependents_1 = 0
        dependents_2 = 0
        dependents_3 = 1

    education_input = request.form['Education']

    if education_input == "Graduate":
        education_graduate = 1
        education_not_graduate = 0
    else:
        education_graduate = 0
        education_not_graduate = 1

    self_employed_input = request.form['Self_Employed']

    if self_employed_input == "Yes":
        self_employed_no = 0
        self_employed_yes = 1
    else:
        self_employed_no = 1
        self_employed_yes = 0

    property_area_input = request.form['Property_Area']

    if property_area_input == "Urban":
        property_area_rural = 0
        property_area_semiurban = 0
        property_area_urban = 1
    elif property_area_input == "Semiurban":
        property_area_rural = 0
        property_area_semiurban = 1
        property_area_urban = 0
    else:
        property_area_rural = 1
        property_area_semiurban = 0
        property_area_urban = 0
    
    predictions_df = pd.DataFrame({"ApplicantIncome": [applicant_income_input],
                                   "CoapplicantIncome": [co_applicant_income_input],
                                   "LoanAmount": [loan_amount_input],
                                   "Loan_Amount_Term": [loan_amount_term_input],
                                   "Credit_History": [credit_history_input],
                                   "Gender_Female": [gender_female],
                                   "Gender_Male": [gender_male],
                                   "Married_No": [married_no],
                                   "Married_Yes": [married_yes],
                                   "Dependents_0": [dependents_0],
                                   "Dependents_1": [dependents_1],
                                   "Dependents_2": [dependents_2],
                                   "Dependents_3+": [dependents_3],
                                   "Education_Graduate": [education_graduate],
                                   "Education_Not Graduate": [education_not_graduate],
                                   "Self_Employed_No": [self_employed_no],
                                   "Self_Employed_Yes": [self_employed_yes],
                                   "Property_Area_Rural": [property_area_rural],
                                   "Property_Area_Semiurban": [property_area_semiurban],
                                   "Property_Area_Urban": [property_area_urban]})


    ## Scale the input data

    x_test_scaled = x_scaler.transform(predictions_df)

    ## Make prediction

    prediction = model.predict_proba(x_test_scaled)
    probability_of_approval = prediction[0][1]

    ## Format probability

    formatted_probability = "{:.2f}%".format(probability_of_approval * 100)

    if probability_of_approval > 0.60:
        page = "approve.html"
    else:
        page = "reject.html"

    return render_template(page, prediction_text='Probability: {}'.format(formatted_probability))

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)