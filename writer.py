#!/usr/bin/python3
import time
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
HOUR = 3600
dir_path = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(dir_path,"Extracted")):
	shutil.rmtree(os.path.join(dir_path,"Extracted"), ignore_errors=True)

os.makedirs(os.path.join(dir_path,"Extracted"))
saved_list = ['N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE','LABEL']
fit_df = pd.DataFrame(columns = saved_list)
#1800 Half Hour
#3600 Full Hour
#proof
#f_fullpath = "./S50ME-C8.2-TII - IGLC Dicle - FWE [2013-12-03 062328 00006].csv"
f_fullpath = "./S50ME-C8.2-TII - IGLC Dicle - FWE [2014-01-02 095027 00010].csv"
df = pd.read_csv(f_fullpath)
#Scaling the values
DATAS = ["FR_eng_meas","RE_HMI_ECUA_020115","RE_gov_idx_meas"]
df.loc[:,'FR_eng_meas'] *= 100/df['FR_eng_meas'].max()
df.loc[:,'RE_HMI_ECUA_020115'] *= 100/df['RE_HMI_ECUA_020115'].max()
df.loc[:,'RE_gov_idx_meas'] *= 100

jack = df[['FR_eng_meas']]
bob = df[['RE_HMI_ECUA_020115']]

measurement = 'RE_HMI_ECUA_020115'
df = df[[measurement]]
df2 = df[df.RE_HMI_ECUA_020115.isnull()]
#ISWS NA VRW ENAN KALUTERO TROPO
iX_missing_Data = []
for i in range(len(df2)):
	iX_missing_Data.append(df2.index[i])
for i in range(len(df)):
	if i in iX_missing_Data:
		df[measurement][i] = 0
segment_size = 180
print("You have chosen to segmentize the data per " + str(segment_size)+ " seconds.")
segment_start = 0
segment_end = segment_size
begin = time.time()
seg_count = 0
#plot_counts = 0
#First Loop through the whole datafile and divide it into segments of information.
for segment in range(0,1600,segment_size):#len(df)-segment_size,segment_size):
	segment_df = df[segment_start:segment_end]
	segment_df = segment_df.reset_index(drop=True)
	windows = 0
	if (segment_df[measurement].mean() == 0):
		pass
	else:
		#plot_counts+=1
		window_size = 60
		window_start = segment_df.index.min()
		window_end = window_start + window_size-1
		#print(segment_df.index.max())
		window_step = 20
		fig = segment_df.plot(color='blue')
		plt.title(r'$Engine \ Speed \ Data$', fontsize=16)
		for window in range(segment_df.index.min(),segment_df.index.max(),window_step):
			#print(segment_df[measurement].iloc[window_start])
			window_df = segment_df[window_start:window_end]
			window_df = window_df.reset_index(drop=True)	
			acel_df = pd.DataFrame()
			for i in range(len(window_df)):
				if i == 0:
					acel_df = acel_df.append([0])
					ace = 0;
					label ="S"
				else:
					acel_df = acel_df.append([window_df[measurement].iloc[i]-window_df[measurement].iloc[i-1]])
					ace = window_df[measurement].iloc[i]-window_df[measurement].iloc[i-1]
			ave=(acel_df[0].mean()*100)
			if ave < 0:
				label ="D"
			elif ave > 0:
				label ="A"
			else:
				label ="S"
			fit_df = fit_df.append({
				'N_MAX': window_df[measurement].max(),
				'N_MIN': window_df[measurement].min(),
				'N_AVE': window_df[measurement].mean(),
				'N_IN' : segment_df[measurement].iloc[window_start],
				'N_OUT': segment_df[measurement].iloc[len(window_df)-1],
				'A_AVE': ave,
				'LABEL': label
				},ignore_index=True)
			plt.text((window_start+window_end)/2, window_df[measurement].iloc[window_df.index.max()], label+'\n'+str(round(ave,2)), bbox=dict(facecolor='red', alpha=0.5))
			plt.axvline(x=window_start+window_df.index.max(), color='red', linestyle=':',linewidth=1, markersize=12)
			plt.text(window_start+window_df.index.max(), plt.ylim()[0], 1, bbox=dict(facecolor='green', alpha=0.5))

			plt.plot(window_start+window_df[measurement].idxmax(),window_df[measurement].max(),'m>') 
			plt.plot(window_start+window_df[measurement].idxmin(),window_df[measurement].min(),'ro')
			plt.plot(window_start,segment_df[measurement].iloc[window_start],'y<')
			plt.plot(window_end,segment_df[measurement].iloc[len(window_df)],'gx')
			plt.plot((window_start+window_end)/2,window_df[measurement].mean(),'r+')
			#plt.text(600, window_df[measurement].mean(), ave, bbox=dict(facecolor='red', alpha=0.5))
			#fig, axs = plt.subplots(nrows=1, ncols=1)
			#fig, axs = plt.plot()
			#reversed_segment_df.plot(ax=axs[1], color='green')
			#fig.suptitle(r'$Governor \ Index \ Data$', fontsize=16)

			#axs.set_title(r"$Profile \  Plot$")
			#axs.set_xlabel(r"$Average \ Times  \ per \ $" + str(segment_size) +r"$ \ secs$")
			#axs.set_ylabel(r"$Governor \ Index \ (\%)$")
			#axs[0].grid(color='black', linestyle='-')

			#axs[1].set_title(r"$Reversed \ Profile \  Plot$")
			#axs[1].set_xlabel(r"$Average \ Times  \ per \ $" + str(segment_size) +r"$ \ secs$")
			#axs[1].set_ylabel(r"$Governor \ Index \ (\%)$")
			#axs[1].grid(color='black', linestyle='-')
			#fig.tight_layout(rect=[0, 0, 1, 0.95])
			
			#print("from",window_start,"to",window_end)
			window_start += window_step
			window_end += window_step
			past_mean_n = window_df[measurement].mean()
	
	segment_start += segment_size
	segment_end += segment_size
	seg_count += 1
	#file_name = os.path.join(dir_path,'Extracted/WIN_'+str(seg_count)+'.csv')
	#csv_df.to_csv(file_name, encoding='utf-8', index=False)
finish = time.time()
fit_df = fit_df.transpose()
print(fit_df)
print("Elapsed time was "+ str(finish-begin))
#print(str(plot_counts)+" plots should have been shown.")
plt.show()
