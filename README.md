# Diabetes Prediction Project

## Overview
This project aims to predict diabetes in patients using various machine learning models. The dataset includes features such as the number of pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, age, smoking status, exercise level, family history of diabetes, and stress level.

## Project Structure
1. **Data Preparation and Exploration**
    - Load dataset
    - Exploratory Data Analysis (EDA)
    - Handle missing values
    - Data visualization
    - Data preprocessing

2. **Model Training and Evaluation**
    - Model Selection: K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Logistic Regression, Naive Bayes
    - Model Evaluation: Accuracy, F1 Score, Classification Report, Confusion Matrix
    - Hyperparameter Tuning with GridSearchCV

3. **Model Deployment**
    - Train the best model on an additional dataset split (exited data)
    - Save the trained model

## Prerequisites
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python libraries:
    - numpy
    - pandas
    - seaborn
    - matplotlib
    - plotly
    - scikit-learn
    - joblib

## Dataset
The dataset used in this project can be found [here](https://docs.google.com/spreadsheets/d/1elzZix69-QC9GJE5t9kYd2b5EjUhLWiCbffk00rkkR8/edit?usp=sharing) 
## How to Run the Code
1. **Mount Google Drive** (if using Google Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Import Libraries**:
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, f1_score
    from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
    import joblib
    import warnings
    import time
    ```

3. **Load Dataset**:
    ```python
    df = pd.read_csv("/content/drive/MyDrive/diabetes_data.csv")
    ```

4. **Exploratory Data Analysis (EDA)**:
    - Display first 5 rows:
      ```python
      df.head()
      ```
    - Data shape, summary information, and descriptive statistics:
      ```python
      print(df.shape)
      df.info()
      print(df["Outcome"].value_counts())
      print(100*df["Outcome"].value_counts()/len(df))
      df.describe()
      df.describe(include="object")
      ```

5. **Handle Missing Values**:
    ```python
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
    df.isnull().sum()
    ```

6. **Replace Missing Values**:
    ```python
    def median_target(var):
        temp = df[df[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

    missing_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']
    for col in missing_columns:
        medians = median_target(col)
        median_dict = medians.set_index('Outcome')[col].to_dict()
        df[col] = df.apply(lambda row: median_dict[row['Outcome']] if pd.isnull(row[col]) else row[col], axis=1)
    print(df[missing_columns].isnull().sum())
    ```

7. **Data Visualization**:
    - Proportion of Diabetes Cases:
      ```python
      fig = px.pie(df, names='Outcome', title='Proportion of Diabetes Cases')
      fig.update_traces(textinfo='percent+label', textfont=dict(size=25))
      fig.update_layout(title={'text': 'Proportion of Diabetes Cases', 'font': {'size': 24, 'family': 'Arial', 'color': 'black'}, 'x': 0.5, 'y': 0.95}, showlegend=False, margin=dict(t=100, b=0, l=0, r=0), hoverlabel=dict(font=dict(size=14)))
      fig.show()
      ```
    - Impact of Categorical Features on the Output:
      ```python
      fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(20,6))
      color = ['#8ebad9', '#ffbe86']
      for i in range(len(category) - 1):
          plt.subplot(1, 4, i+1)
          ax = sns.countplot(x=category[i], data=categorical_values, hue="Outcome", palette=color, edgecolor='black')
      plt.suptitle('Impact of Categorical Features on the Output')
      plt.show()
      ```
    - Distribution and Boxplot of Features:
      ```python
      plt.figure(figsize=(15, 10))
      for i, col in enumerate(df.columns, 1):
          plt.subplot(4, 4, i)
          plt.title(f"Distribution of {col}")
          sns.histplot(df[col], kde=True)
          plt.tight_layout()
      plt.show()
      ```

8. **Data Preprocessing**:
    - Check for outliers and handle them:
      ```python
      def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
          quartile1 = dataframe[col_name].quantile(q1)
          quartile3 = dataframe[col_name].quantile(q3)
          interquantile_range = quartile3 - quartile1
          up_limit = quartile3 + 1.5 * interquantile_range
          low_limit = quartile1 - 1.5 * interquantile_range
          return low_limit, up_limit

      def check_outlier(dataframe, col_name):
          if dataframe[col_name].dtype.kind in 'bifc':
              low_limit, up_limit = outlier_thresholds(dataframe, col_name)
              if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
                  return True
          return False

      cols = [col for col in df.columns if df[col].dtype.kind in 'bifc']
      for col in cols:
          print(col, check_outlier(df, col))

      def replace_with_thresholds(dataframe, variable):
          low_limit, up_limit = outlier_thresholds(dataframe, variable)
          dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
          dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

      replace_with_thresholds(df, "Insulin")
      for col in cols:
          print(col, check_outlier(df, col))
      ```

9. **Data Scaling and Encoding**:
    - Scale numeric data:
      ```python
      for col in cols:
          df[col] = RobustScaler().fit_transform(df[[col]])
      ```
    - Encode categorical data:
      ```python
      le = LabelEncoder()
      for col in df.select_dtypes(include='object'):
          df[col] = le.fit_transform(df[col])
      ```

10. **Split Data**:
    ```python
    split = int(df.shape[0] * 0.8)
    train_test = df.iloc[:split, :]
    exited = df.iloc[split:, :]

    X_train_test = train_test.drop(columns=['Outcome'])
    y_train_test = train_test['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=36)
    ```

11. **Model Selection and Evaluation**:
    - Apply KNN, SVC, Logistic Regression, Naive Bayes models, and evaluate their performance.

12. **Hyperparameter Tuning**:
    ```python
    models_param_grids = {
        'KNN': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}},
        'SVC': {'model': SVC(), 'params': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}},
        'Logistic Regression': {'model': LogisticRegression(random_state=0), 'params': {'C': [0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'max_iter': [100, 200, 300]}},
        'Naive Bayes': {'model': GaussianNB(), 'params': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}}
    }

    for model_name, mp in models_param_grids.items():
        grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(f'Best parameters for {model_name}: {grid.best_params_}')
        print(f'Best accuracy score for {model_name}: {grid.best_score_}')

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        print(f'Accuracy for {model_name} on test data: {accuracy_score(y_test, y_pred)}')
        print(f'F1 Score for {model_name} on test data: {f1_score(y_test, y_pred)}')
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        if model_name == 'Logistic Regression':
            roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label=f'{model_name} ROC Curve (area = {roc_auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.show()
    ```

13. **Model Deployment**:
    ```python
    final_model = LogisticRegression(random_state=0, C=0.1, max_iter=100, solver='lbfgs')
    X_exited = exited.drop(columns=['Outcome'])
    y_exited = exited['Outcome']
    final_model.fit(X_exited, y_exited)

    joblib.dump(final_model, 'diabetes_prediction_model.pkl')
    ```

## Conclusion
This project demonstrates the process of predicting diabetes using machine learning models. It covers data preprocessing, model selection, hyperparameter tuning, evaluation, and deployment. The Logistic Regression model with specific hyperparameters is chosen as the final model based on its performance metrics. The model is then trained on an additional dataset and saved for future use.


