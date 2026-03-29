# Bank Fraud Transitions Detection Model

Bank Fraud Transitions Detection Model is a machine learning project that helps identify fraudulent bank transactions. It uses a trained Logistic Regression model to analyze transaction details and predict whether a transaction is safe or fraudulent. The project also includes a simple and interactive web application built with Streamlit, where users can enter transaction details and get instant predictions.

# Project Overview

Online transactions are increasing every day, and the risk of fraud is also growing. This project aims to address this problem by using data analysis and machine learning techniques to detect suspicious transactions. The model is trained on transaction data and learns important patterns such as unusual transaction amounts, sudden changes in account balances, and specific transaction types that are more likely to be fraud.

# Features

This project can predict whether a transaction is fraud or not fraud in a fast and efficient way. It provides a simple and clean Streamlit web interface where users can easily input transaction details. The system supports multiple transaction types such as PAYMENT, TRANSFER, CASH_OUT, and DEPOSIT. It uses real transaction features like amount and account balances to make predictions and delivers results instantly, making it useful for quick analysis.

# Technologies Used

The project is developed using Python as the main programming language. It uses Pandas and NumPy for data processing and handling, while Matplotlib and Seaborn are used for data visualization. Scikit-learn is used to build and train the machine learning model, and Streamlit is used to create the interactive web application.

# Machine Learning Model

The model used in this project is Logistic Regression, which is suitable for classification problems like fraud detection. Since fraud datasets are usually imbalanced, the model handles this issue using class weighting. The data is preprocessed using techniques such as Standard Scaling for numerical features and One-Hot Encoding for categorical features like transaction type. After training, the model is saved using Joblib and later loaded into the Streamlit application for real-time predictions.

# Project Structure

The project consists of multiple files that work together. The analysis_model.ipynb file contains the data analysis and model training process. The fraud_detection.py file is used to run the Streamlit web application. The trained model is saved as fraud_detection_pipeline.pkl, which is used for making predictions. The dataset file, if included, is named AIML Dataset.csv, and the README.md file contains all the project documentation.

# How to Run the Project

To run this project, first clone the repository from GitHub to your local system and navigate into the project folder. After that, install the required libraries using pip. If a requirements file is available, you can install all dependencies using a single command; otherwise, you can manually install libraries such as pandas, numpy, scikit-learn, streamlit, matplotlib, and seaborn. Once everything is installed, run the Streamlit application using the command streamlit run fraud_detection.py, which will open the app in your browser.

# How to Use

After opening the application in your browser, you need to enter the transaction details such as transaction type, amount, sender’s balance before and after the transaction, and receiver’s balance before and after the transaction. Once all the details are filled in, click on the Predict button. The application will then display the result, indicating whether the transaction is fraud or not, along with a clear message if the transaction is risky.

# Example

For example, if a transaction shows unusual behavior such as the sender’s balance becoming zero immediately after a transfer, the model may classify it as a fraudulent transaction based on patterns it has learned during training.

# Note

This project is mainly created for learning and demonstration purposes. The predictions are based on trained data and may not always be completely accurate. Therefore, it is not recommended to use this system in real banking environments without further improvements and testing.

# Future Improvements

In the future, this project can be improved by using more advanced machine learning models such as Random Forest or XGBoost. The accuracy can be enhanced by better feature engineering and adding more meaningful data features. The application can also be upgraded to show fraud probability scores and deployed online to make it accessible to more users.

# Author

A. Kumar
