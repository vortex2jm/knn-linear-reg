import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from modules.knn import knn_regressor
from modules.lreg import linear_regressor
from modules.intervalo_confianca import intervalo_de_confianca

def main():
    # Data reading
    df = pd.read_csv("./data/Automobile.csv")

    # Data processing
    df = df.drop(columns=["name", "origin"]).dropna()
    df['cylinders'] = df['cylinders'].astype(np.float64)
    df['weight'] = df['weight'].astype(np.float64)
    df['model_year'] = df['model_year'].astype(np.float64)

    inputs = df.iloc[:, 1:]
    output = df.mpg

    # Mean absolute errors
    final_mae_knn = []
    final_mae_lr = []

    for i in range(30):
        # KNN regressor ==================================== #
        k_mae = knn_regressor(inputs, output)
        winner_mae = k_mae[k_mae.index(min(k_mae))]
        final_mae_knn.append(winner_mae)
        
        #Linear regressor ================================== #
        linear_mae = linear_regressor(inputs, output)
        final_mae_lr.append(linear_mae)

    print("Avarage MAE [KNN]: ", sum(final_mae_knn)/len(final_mae_knn))
    print('Avarage MAE [LR]: ', sum(final_mae_lr)/len(final_mae_lr))

    print("Intervalo de confiança MAE [KNN, 95%]: ", intervalo_de_confianca(final_mae_knn))
    print('Intervalo de confiança MAE [LR, 95%]: ', intervalo_de_confianca(final_mae_lr))

    print("media real: ", sum(output)/len(output))
    
if __name__ == "__main__":
   main()
