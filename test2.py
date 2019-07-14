import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TREND_FEATURES = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']

def labelMaker(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

X_data = pd.read_csv("train.csv",usecols=TREND_FEATURES)
X_data = X_data.astype({'LABEL': int})
#X_data = normalizeDataFrame(X_data)

X_train, X_test = train_test_split(X_data, test_size=0.3, shuffle=False)

Y_train = X_train[['LABEL']]
Y_test = X_test[['LABEL']]
Y_train = labelMaker(Y_train.values,5)
print(type(Y_train))
