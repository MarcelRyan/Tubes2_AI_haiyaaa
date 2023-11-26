import numpy as np
import pandas as pd

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def neighbour(train_data, test_instance, k):
    distances = np.apply_along_axis(lambda x: euclidean_distance(x, test_instance), 1, train_data)
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k]

def knn(data, test_instance, k):
    price_range = data['price_range'].values
    data_train = data.drop('price_range', axis=1).values
    neighbours_idx = neighbour(data_train, test_instance, k)
    neighbour_price_range = price_range[neighbours_idx]
    return pd.Series(neighbour_price_range).mode().iloc[0]

def knn_prediction(data_train, data_validation, k):
    predictions = []
    for i in range(len(data_validation)):
        test_instance = data_validation.iloc[i, :].values
        predictions.append(knn(data_train, test_instance, k))
    return np.array(predictions)



