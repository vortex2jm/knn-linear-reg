from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def knn_regressor(inputs, output):
  k_mae = []
  for k in range(10):
    # Setting testing and training data
    x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.1, train_size=0.9)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # Instantiating KNN regressor
    knn_regressor = KNeighborsRegressor(n_neighbors=k+1)
    knn_regressor.fit(x_train, y_train)

    # Defining predictions
    y_pred = knn_regressor.predict(x_test)

    # Getting mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    k_mae.append(mae)
    
  return k_mae
