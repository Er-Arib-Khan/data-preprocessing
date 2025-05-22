from sklearn.impute import SimpleImputer
import numpy as np
data = np.array([
    [1,2,np.nan],
    [4,np.nan,6],
    [np.nan,8,9]
])
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(data)
print(data)
print(imputed_data)