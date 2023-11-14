import numpy as np
import pandas as pd

# Sample dataset
data = {
    'NumericFeature1': [1, 2, 3, 4, 5],
    'NumericFeature2': [2.0, 4.0, 6.0, 8.0, 10.0],
    'Label': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

def euclidean_distance(x1, x2):
    sum=0
    for i in range(len(x1)):
        sum += (float(x1[i]) - float(x2[i])) ** 2
    return np.sqrt(sum)

def knn(train_data, test_instance, k):
    pair = []

    for idx,data in train_data.iterrows():
        # print(idx,' ',data.values)
        distance = euclidean_distance(data.values, test_instance)
        print(distance)
        # print(data.values[0])
        # print(data.values[1])
        # print(data.values[2])
        # print('\n')
        pair.append((idx, distance))

    print(pair)
    sorted_pair = sorted(pair, key=lambda x: x[1])
    print(sorted_pair)

    # return k nearest neighbours
    return sorted_pair[:k]

test_instance = np.array([3, 6.0, 0])
print(knn(df, test_instance, 3))


# x1 = [1,2,3]
# x2 = [0,0,0]
# # print(x1-x2)

# print(euclidean_distance(x1, x2))
