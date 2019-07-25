import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("train.csv")
data_0 = data.loc[data['LABEL'] == 0]
data_1 = data.loc[data['LABEL'] == 1]
data_2 = data.loc[data['LABEL'] == 2]
data_3 = data.loc[data['LABEL'] == 3]
data_4 = data.loc[data['LABEL'] == 4]
data_5 = data.loc[data['LABEL'] == 5]
data_6 = data.loc[data['LABEL'] == 6]

data_0 = data_0.loc[data['LABEL'] == 0]
data_1 = data_1.loc[data['LABEL'] == 1]
data_2 = data_2.loc[data['LABEL'] == 2]
data_3 = data_3.loc[data['LABEL'] == 3]
data_4 = data_4.loc[data['LABEL'] == 4]
data_5 = data_5.loc[data['LABEL'] == 5]
data_6 = data_6.loc[data['LABEL'] == 6]
print(data_0.describe().T)
print(data_1.describe().T)
print(data_2.describe().T)
print(data_3.describe().T)
print(data_4.describe().T)
print(data_5.describe().T)
print(data_6.describe().T)
sns.pairplot(data[["LABEL","N_MAX", "N_AVE"]], diag_kind="kde",hue='LABEL')




#data.plot()
plt.show()
