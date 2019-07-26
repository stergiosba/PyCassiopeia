import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_pickle("../swap_corrected_templates_soft_dtw_clusters7_gamma1.pkl")
sample = pd.read_csv("sample_3.csv")
data = data.head(1600)
    
data.plot()
sample.plot()
plt.show()
