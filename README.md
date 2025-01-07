# Building an Intelligent Loan Default Prediction System
## Project Overview: Building an Intelligent Loan Default Prediction System

**Problem Statement**
In the world of lending, one of the biggest challenges banks and financial institutions face is predicting whether a borrower will default on their loan. Getting this wrong can be costly:
•	If we predict someone will pay back but they default, the bank loses money.
•	If we predict someone will default but they would have paid back, we lose a good customer.

**Our Dataset**
We're working with synthetic loan data that includes 9,500 loans. For each loan, we have 14 different pieces of information:
1.	'credit.policy': Whether the customer meets credit underwriting criteria
2.	'purpose': The purpose of the loan (debt consolidation, education, etc.)
3.	'int.rate': Interest rate on the loan
4.	'installment': Monthly installment amount
5.	log.annual.inc': Natural log of the borrower's annual income
6.	'dti': Debt-to-Income ratio
7.	'fico': FICO credit score
8.	'days.with.cr.line': Number of days with credit line
9.	'revol.bal': Revolving balance
10.	'revol.util': 'Revolving line utilization rate',
11.	'inq.last.6mths': Number of inquiries in last 6 months
12.	'delinq.2yrs': Number of delinquencies in past 2 years
13.	'pub.rec': Number of public records
14.	'not.fully.paid': Whether the loan was not paid back (our target variable)


**Project Goals**
Primary Goal: Build a machine learning model that can accurately predict loan defaults
•	Focus on identifying risky loans (those that won't be paid back)
•	Handle the challenge of imbalanced data (typically fewer defaults than non-defaults)
Secondary Goals:
•	Create clear visualizations to understand what makes a loan risky
•	Build an interactive web app where users can input loan information and get predictions

**Success Metrics**
We'll measure our success using metrics that matter for imbalanced classification:

•	Precision: How many of our predicted defaults are actual defaults?
•	Recall: What percentage of actual defaults can we catch?
•	F1-Score: A balance between precision and recall
•	ROC-AUC: How well can our model distinguish between classes?


