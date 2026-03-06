# Heart Disease Prediction

Predicting heart disease using logistic regression and a neural network.

## Dataset

Heart disease dataset from kaggle (UCI). 302 patient records, 13 features like age, cholesterol, blood pressure etc. Target is 1 (disease) or 0 (no disease).

## What was done

- basic EDA - checked stats, class balance, correlation heatmap
- scaled features with StandardScaler
- trained logistic regression with GridSearchCV to find best hyperparameters
- built a simple ANN with keras (64 -> dropout -> 32 -> 1)
- compared both models on accuracy, precision, recall, f1, auc
- plotted confusion matrices and ROC curves

## How to run

```
pip install -r requirements.txt
```

put heart.csv in the same folder then run the notebook.

## Tools

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras
