from sklearn.preprocessing import MinMaxScaler
import numpy as np
data = np.array([[10,20],
                 [15,30],
                 [25,45],
                 [30,60]])
scaler = MinMaxScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
print(data)
print(scaled_data)
