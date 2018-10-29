
Machine Learning Course - CS-433 Project 1
******************************************
Authors: Matthias Minder, Zora Oswald, Silvan Stettler


Execution: Run the run.py script in the folder src
The root folder (where this readme is located) must contain a folder all which contains the two data files train.csv and test.csv
The submission SVM_on_imputed_poly.csv is created in the folder submission in the root folder
In order not to have to reexecute the whole code every time, an intermediate result (with the imputation of the missing values) is saved in a folder imputed in the root folder

The code includes three files of methods:
- proj1_helpers.py: The unmodified file provided by the course instructors
- implementations.py: File containing different classification tools (linear regression, logistic regression, support vector machine)
- custom_helpers.py: File containing a data normalization function and a cross-validation algorithm
