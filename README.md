# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.

2.Load Dataset: Load the dataset containing car prices and relevant features.

3.Data Preprocessing: Handle missing values and perform feature selection if necessary.

4.Split Data: Split the dataset into training and testing sets.

5.Train Model: Create a linear regression model and fit it to the training data.

6.Make Predictions: Use the model to make predictions on the test set.

7.Evaluate Model: Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.

8.Check Assumptions: Plot residuals to check for homoscedasticity, normality, and linearity.

9.Output Results: Display the predictions and evaluation metrics.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Yugabharathi T
RegisterNumber:  212224040375
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment (2).csv')


x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train_scaled,y_train)

y_pred=model.predict(x_test_scaled)
print("="*50)
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns,model.coef_):
    print(f" {feature:>12}: {coef:>10.2f}")
print(f"{'Intercept': >12}: {model.intercept_:>10.2f}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
print("-"*50)

#1. Linearity check
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()


#2. Independence (Durbin-Watson)
residuals= y_test- y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}", "\n(Values close to 2 indicate no autocorrelation)")


#3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()



#4. Normality of residuals
fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="636" height="291" alt="image" src="https://github.com/user-attachments/assets/bc646dcf-b5ab-427e-9c07-b8a7359ac2ba" />
<img width="1187" height="595" alt="image" src="https://github.com/user-attachments/assets/1ec4f5dd-22f7-4273-9c2c-b386bc60f83c" />
<img width="792" height="394" alt="image" src="https://github.com/user-attachments/assets/08303ac2-c7e2-426a-b53b-6b5848cb7ac9" />
<img width="778" height="316" alt="image" src="https://github.com/user-attachments/assets/58a0b420-f531-49a1-9b94-617aa5019a06" />







## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
