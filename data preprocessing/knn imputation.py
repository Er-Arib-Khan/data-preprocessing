from sklearn.impute import KNNImputer
import numpy as np
data = np.array([
    [1,2,np.nan],
    [4,np.nan,6],
    [np.nan,8,9]
])
knn_imputer = KNNImputer(n_neighbors=2)
imputed_data = knn_imputer.fit_transform(data)
print(data)
print(imputed_data)