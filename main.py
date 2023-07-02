
def main():
    import numpy as np
    import pandas as pd

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error

    #Leitura dos dados
    df = pd.read_csv("./data/Automobile.csv")

    #Tratamento de dados
    df = df.drop(columns=["name", "origin"]).dropna()
    df['cylinders'] = df['cylinders'].astype(np.float64)
    df['weight'] = df['weight'].astype(np.float64)
    df['model_year'] = df['model_year'].astype(np.float64)

    inputs = df.iloc[:, 1:]
    output = df.mpg

    final_mae = []

    for i in range(30):

        k_mae = []
        for k in range(10):
            x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.1, train_size=0.9)

            scaler = StandardScaler()

            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)

            knn_regressor = KNeighborsRegressor(n_neighbors=k+1)

            knn_regressor.fit(x_train, y_train)

            y_pred = knn_regressor.predict(x_test)

            mae = mean_absolute_error(y_test, y_pred)

            k_mae.append(mae)

        winner_mae = k_mae[k_mae.index(min(k_mae))]
        final_mae.append(winner_mae)

    print("valores finais de MAE: \n", final_mae)

    print("MAE médio: ", sum(final_mae)/len(final_mae))




    
if __name__ == "__main__":
   main()
