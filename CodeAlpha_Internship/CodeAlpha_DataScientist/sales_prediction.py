import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Note: You would usually load this from 'advertising.csv'
# Creating a sample structure for demonstration:
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)

# 1. Feature Selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 4. Prediction & Performance
y_pred = regressor.predict(X_test)
print(f"R-Squared Score: {r2_score(y_test, y_pred):.4f}")

# 5. Visualization
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()