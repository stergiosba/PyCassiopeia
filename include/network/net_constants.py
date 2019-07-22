'''Tags used for Networks loading and creation.

'''
CREATE = 'create'
LOAD = 'load'

'''Common tags used for graphs/figures in Networks and saving in models.

'''
TRAINING = 'train'
TESTING = 'test'
BUILD = 'build'

'''Common tags used for directories in Networks and Window Models.

'''
CYCLES = 'Cycles'
TREND = 'Trend'
SOC = 'Battery'
WENG = 'Engine'

'''Common tags used for different Networks.

'''
NN_CYCLES = "NN_EC"
#CYCLES_FEATURES = ['LABEL','N_MAX','A_MAX','A_AVE','A_STD','D_MAX','D_AVE','P_N_030','P_N_3050','P_N_5080','P_N_80100','N_AVE']#,'P_D_12','P_D_23']
CYCLES_FEATURES = ['LABEL','N_MAX','N_AVE','A_MAX','A_AVE','A_STD','D_MAX','D_AVE','P_N_030','P_N_3050','P_N_5070','P_N_70100']#,'P_D_12','P_D_23']
CYCLES_OUTPUTS = 7

NN_TREND ="NN_DT"
TREND_FEATURES = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']
#TREND_OUTPUTS = '3'
TREND_OUTPUTS = 5

NN_BATTERY = "NN_BAT"
BATTERY_FEATURES = ['TREND','SPEED','POWER','SOC']
BATTERY_OUTPUTS = 1

NN_WENG = "NN_NENG"
WENG_FEATURES = ['SPEED','POWER','SOC']
WENG_OUTPUTS = 1





