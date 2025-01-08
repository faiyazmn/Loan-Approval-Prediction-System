import gradio as gr
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("loan_classifier.joblib")
scaler = joblib.load("std_scaler.joblib")

def predict_loan_status(
   int_rate, installment, log_annual_inc, dti, fico, revol_bal, 
   revol_util, inq_last_6mths, delinq_2yrs, pub_rec, 
   installment_to_income_ratio, credit_history
):
   # Create input array
   input_array = np.array([[
       int_rate, installment, log_annual_inc, dti, fico, revol_bal,
       revol_util, inq_last_6mths, delinq_2yrs, pub_rec,
       installment_to_income_ratio, credit_history
   ]])
   
   # Scale input and predict
   scaled_array = scaler.transform(input_array)
   prediction_proba = model.predict_proba(scaled_array)[0]
   
   return {
       "Loan Fully Paid": float(prediction_proba[0]),
       "Loan Not Fully Paid": float(prediction_proba[1])
   }

# Interface inputs
inputs = [
   gr.Slider(0.06, 0.23, step=0.01, label="Interest Rate"),
   gr.Slider(100, 950, step=10, label="Installment"),
   gr.Slider(7, 15, step=0.1, label="Log Annual Income"),
   gr.Slider(0, 40, step=1, label="DTI Ratio"),
   gr.Slider(600, 850, step=1, label="FICO Score"),
   gr.Slider(0, 120000, step=1000, label="Revolving Balance"),
   gr.Slider(0, 120, step=1, label="Revolving Utilization"),
   gr.Slider(0, 10, step=1, label="Inquiries in Last 6 Months"),
   gr.Slider(0, 20, step=1, label="Delinquencies in Last 2 Years"),
   gr.Slider(0, 10, step=1, label="Public Records"),
   gr.Slider(0, 5, step=0.1, label="Installment to Income Ratio"),
   gr.Slider(0, 1, step=0.01, label="Credit History")
]

interface = gr.Interface(
   fn=predict_loan_status,
   inputs=inputs,
   outputs=gr.Label(num_top_classes=2),
   title="Loan Default Predictor",
   description="Enter loan applicant details to predict default probability"
)

interface.launch(share=True)