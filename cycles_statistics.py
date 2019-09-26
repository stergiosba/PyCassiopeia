import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("swap_corrected_templates_soft_dtw_clusters7_gamma1.csv")
for cycle in data.columns:
    cycle_df = data[cycle]
    cycle_df=cycle_df.head(1500)
    print(cycle_df.describe())

#sns.pairplot(data[["LABEL","N_MAX", "N_AVE"]], diag_kind="kde",hue='LABEL')

#data.plot()
#plt.show()
