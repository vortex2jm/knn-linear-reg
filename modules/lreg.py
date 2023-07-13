from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def linear_regressor(inputs, output, degree=2):
  x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.1, train_size=0.9)
  
  poly_features = PolynomialFeatures(degree=degree)
  x_poly_train = poly_features.fit_transform(x_train)
  x_poly_test = poly_features.fit_transform(x_test)
  
  linear_regressor = LinearRegression();
  linear_regressor.fit(x_poly_train, y_train)
  linear_y_pred = linear_regressor.predict(x_poly_test)
  linear_mae = mean_absolute_error(y_test, linear_y_pred)
  return linear_mae
