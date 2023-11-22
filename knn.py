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
        # sum += (float(x1[i]) - float(x2[i])) ** 2
        sum += (float(x1.iloc[i]) - float(x2.iloc[i])) ** 2
    return np.sqrt(sum)

def neighbour(train_data, test_instance, k):
    pair = []

    # for idx,data in train_data.iterrows():
    #     distance = euclidean_distance(data.values, test_instance)
    #     pair.append((idx, distance))
    
    for idx in range(len(train_data)):
        distance = euclidean_distance(train_data.iloc[idx,:], test_instance)
        pair.append((idx, distance))
    # print(pair)
    sorted_pair = sorted(pair, key=lambda x: x[1])
    # print(sorted_pair)

    # return k nearest neighbours
    return sorted_pair[:k]

def knn(data, test_instance, k):
    # anggap train data df semuanya, slicing disini
    price_range = data.iloc[:, data.columns == 'price_range']
    
    data_train = data.iloc[:,df.columns != 'price_range']
    neighbours = neighbour(data_train, test_instance, k)
    neighbours_idx = [neighbour[0] for neighbour in neighbours]
    neighbour_price_range = price_range.filter(items=neighbours_idx, axis=0)
    print(neighbour_price_range)
    return neighbour_price_range.mode().iloc[0]

df = pd.read_csv('data_train.csv')
# print(df.iloc[0,df.columns != 'price_range'])
# # print(df.iloc[0])
columns =['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
test_instance = pd.Series([700, 0, 1.1, 0, 13, 1, 35, 0.5, 117, 6, 10, 512, 1001, 2000, 11, 5, 10, 0, 1, 1])
# test_instance = df.iloc[0,df.columns != 'price_range'] # without price_range col
data_train = df.iloc[:,df.columns != 'price_range']

print(knn(df, test_instance, 100))
# print(df.iloc[0])
# print(df.iloc[436])
# print(df.iloc[785])
# test_instance = np.array([3, 6.0, 0])
# neighbours = neighbour(data_train, test_instance, 3)
# filter = [neighbour[0] for neighbour in neighbours]
# print(filter)
# print(df.iloc[])


# x1 = [1,2,3]
# x2 = [0,0,0]
# # print(x1-x2)

# print(euclidean_distance(x1, x2))
