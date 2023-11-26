import numpy as np
import pandas as pd

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def neighbour(train_data, test_instance, k):
    # Iteration along axis (all data trains) to calculate euclidean distance between test instance and the whole train data
    distances = np.apply_along_axis(lambda x: euclidean_distance(x, test_instance), 1, train_data)
    # Sorting all distances to collect the shortests distances with k amount.
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k]

def knn(data, test_instance, k):
    # Get target attribute
    price_range = data['price_range'].values
    # Data train without the target attribute
    data_train = data.drop('price_range', axis=1).values

    # Getting k amount of shortest distance values (just the index)
    neighbours_idx = neighbour(data_train, test_instance, k)
    neighbour_price_range = price_range[neighbours_idx]

    # returning mode of most frequent price range value collected from neighbors.
    return pd.Series(neighbour_price_range).mode().iloc[0]

def knn_prediction(data_train, data_validation, k):
    # Iteration for multiple instances
    predictions = []
    for i in range(len(data_validation)):
        test_instance = data_validation.iloc[i, :].values
        predictions.append(knn(data_train, test_instance, k))
    return np.array(predictions)



