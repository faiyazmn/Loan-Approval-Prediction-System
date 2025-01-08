# Building a Loan Approval Prediction System

**Problem Statement**
In the world of lending, one of the biggest challenges banks and financial institutions face is predicting whether a borrower will default on their loan. Getting this wrong can be costly:
*	If we predict someone will pay back but they default, the bank loses money.
*	If we predict someone will default but they would have paid back, we lose a good customer.

**Our Dataset**
We're working with synthetic loan data that includes 9,578 loans. For each loan, we have 14 different pieces of information:
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
*	Focus on identifying risky loans (those that won't be paid back)
*	Handle the challenge of imbalanced data (typically fewer defaults than non-defaults)
Secondary Goals:
*	Create clear visualizations to understand what makes a loan risky
*	Build an interactive web app where users can input loan information and get predictions

**Success Metrics**
We'll measure our success using metrics that matter for imbalanced classification:

*	Precision: How many of our predicted defaults are actual defaults?
*	Recall: What percentage of actual defaults can we catch?
*	F1-Score: A balance between precision and recall
*	ROC-AUC: How well can our model distinguish between classes?

## Data Exploration and Preparation
During our analysis of the loan data, we discovered several important patterns:
*	About 20% of loans in our dataset were not fully paid, highlighting the importance of accurate default prediction
*	FICO scores showed a strong relationship with loan defaults - borrowers with higher scores were less likely to default
*	Higher interest rates were associated with increased default risk
*	Debt consolidation was the most common loan purpose

To improve our predictions, we created two additional metrics:
*	Installment-to-income ratio: Helps assess if monthly payments are manageable given the borrower's income
*	Credit history score: Combines past delinquencies and public records with FICO score

## Model Development
We tested several machine learning approaches to find the best method for predicting loan defaults. The Random Forest model emerged as our top performer, achieving:
*	89% overall accuracy
*	Strong ability to identify both good loans and potential defaults
*	Reliable predictions across different loan types and amounts

Key factors that our model identified as most important for predicting defaults:
1.	FICO Credit score
2.	Debt-to-income ratio
3.	Interest rate
4.	Payment-to-income ratio
5.	Credit history

## Web Application Development
To make our model practical and user-friendly, we developed an interactive web application that:
*	Allows loan officers to input applicant information
*	Provides instant risk assessment
*	Shows the probability of loan default
*	Features an intuitive interface with slider controls for each input

The application is now live and can be accessed through Hugging Face Spaces, https://huggingface.co/spaces/faiyazmn/loan-app  


## Business Impact
This loan default prediction system offers several benefits:
*	More accurate risk assessment
*	Faster loan processing
*	Consistent evaluation criteria
*	Potential reduction in default rates
*	Better customer experience for qualified borrowers


## Conclusion
Our loan default prediction system provides a robust, data-driven approach to loan risk assessment. By combining advanced analytics with an easy-to-use interface, we've created a practical tool that can help improve lending decisions while maintaining accessibility for all users.



