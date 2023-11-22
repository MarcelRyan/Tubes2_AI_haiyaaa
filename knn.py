import numpy as np
import pandas as pd

def euclidean_distance(x1, x2):
    sum=0
    for i in range(len(x1)):
        # sum += (float(x1[i]) - float(x2[i])) ** 2
        sum += (float(x1.iloc[i]) - float(x2.iloc[i])) ** 2
    return np.sqrt(sum)

def neighbour(train_data, test_instance, k):
    pair = []
    
    for idx in range(len(train_data)):
        distance = euclidean_distance(train_data.iloc[idx,:], test_instance)
        pair.append((idx, distance))
    sorted_pair = sorted(pair, key=lambda x: x[1])

    # return k nearest neighbours
    return sorted_pair[:k]

def knn(data, test_instance, k):
    # anggap train data df semuanya, slicing disini
    price_range = data.iloc[:, data.columns == 'price_range']
    
    data_train = data.iloc[:,data.columns != 'price_range']
    neighbours = neighbour(data_train, test_instance, k)
    neighbours_idx = [neighbour[0] for neighbour in neighbours]
    neighbour_price_range = price_range.filter(items=neighbours_idx, axis=0)
    return neighbour_price_range.mode().iloc[0].values[0]

def knn_prediction(data_train, data_validation, k):
    prediction = []
    for i in range (len(data_validation)):
        prediction.append(knn(data_train, data_validation.iloc[i,:], k))
    return np.array(prediction)



