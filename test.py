from knn import * 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# pake algo sendiri
df = pd.read_csv('data_train.csv')
df2 = pd.read_csv('data_validation.csv')
# print(df.iloc[0,df.columns != 'price_range'])
# # print(df.iloc[0])
columns =['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
test_instance = pd.Series([700, 0, 1.1, 0, 13, 1, 35, 0.5, 117, 6, 10, 512, 1001, 2000, 11, 5, 10, 0, 1, 1])
# test_instance = df.iloc[0,df.columns != 'price_range'] # without price_range col
data_train = df.iloc[:,df.columns != 'price_range']

# print(knn(df, test_instance, 100))


# pake library
X_train = df.drop(columns=['price_range'], axis=1)
y_train = df['price_range']

X_test = df2.drop(columns=['price_range'], axis=1)
y_test = df2['price_range']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Prediction: ", y_pred)
print("Accuracy:", accuracy)

print("____________________________________ bikin sendiri:")
y_pred2 = knn_prediction(df, X_test, 3) 
accuracy2 = accuracy_score(y_test, y_pred2)
print("Prediction2: ", y_pred2)
print("Accuracy2: ", accuracy2)

