from knn import * 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# pake algo sendiri
df = pd.read_csv('./data/data_train.csv')
df2 = pd.read_csv('./data/data_validation.csv')

#CLEANING

# Show outliers (numeric)
numeric_cols = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h']
column_name = 'px_height'

# Replace values less than 217 with 217 in the specified column using numpy
df[column_name] = np.where(df[column_name] < 217, 217, df[column_name])

column_name = "sc_w"

df[column_name] = np.where(df[column_name] < 2.5, 2.5, df[column_name])

X_train = df.drop(columns=['m_dep', 'talk_time', 'clock_speed', 'n_cores', 'pc', 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'], axis=1)
y_train = df['price_range']
# X_train = df.drop(columns=['price_range'], axis=1)
# y_train = df['price_range']
X_test = df2.drop(columns=['price_range', 'm_dep', 'talk_time', 'clock_speed', 'n_cores', 'pc', 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'], axis=1)
y_test = df2['price_range']

# X_test = df2.drop(columns=['price_range'], axis=1)
# y_test = df2['price_range']
# knn = KNeighborsClassifier(n_neighbors=19) # 19
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print("Prediction: ", y_pred)
# print("Accuracy:", accuracy)

print("____________________________________ bikin sendiri:")
y_pred2 = knn_prediction(X_train, X_test, 1) 
accuracy2 = accuracy_score(y_test, y_pred2)
print("Prediction2: ", y_pred2)
print("Accuracy2: ", accuracy2)

