from knn import * 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# pake algo sendiri
df = pd.read_csv('./data/data_initial_train.csv')
df2 = pd.read_csv('./data/test.csv')

#CLEANING

# Show outliers (numeric)
numeric_cols = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h']
column_name = 'px_height'

# Replace values less than 217 with 217 in the specified column using numpy
df[column_name] = np.where(df[column_name] < 217, 217, df[column_name])

column_name = "sc_w"

df[column_name] = np.where(df[column_name] < 2.82, 2.82, df[column_name])

X_train = df.drop(columns=['m_dep', 'talk_time', 'clock_speed', 'n_cores', 'pc', 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'], axis=1)
y_train = df['price_range']
# X_train = df.drop(columns=['price_range'], axis=1)
# y_train = df['price_range']
ids = df2['id']
X_test = df2.drop(columns=['id', 'm_dep', 'talk_time', 'clock_speed', 'n_cores', 'pc', 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'], axis=1)

print("____________________________________ bikin sendiri:")
y_pred2 = knn_prediction(X_train, X_test, 1) 
df_submission = pd.DataFrame(data={"id": ids, "price_range": y_pred2})
print(df_submission)
df_submission.to_csv("submission.csv", index=False)

