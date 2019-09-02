'''Tags used for Networks loading and creation.

'''
CREATE = 'create'
LOAD = 'load'
CLASSIFIERS = 'models'
CONTROLLERS = 'controllers'
NNCONTROLLER = 'NNCON'

'''Common tags used for graphs/figures in Networks and saving in models.

'''
TRAINING = 'train'
TESTING = 'test'
BUILD = 'build'
SIMULATION = 'simulation'

'''Common tags used for directories in Networks and Window Models.

'''
CYCLES = 'Cycles'
TREND = 'Trend'
MOTOR = 'Motor'
ENGINE = 'Engine'
CYCLE = 'Cycle'
SIMULATIONS = 'Simulations'

'''Common tags used for different Networks.

'''
TEST_FEATURES = ['Tice','Tel','SOC','Vship']
NN_CYCLES = "NN_EC"
#CYCLES_FEATURES = ['LABEL','N_MAX','A_MAX','A_AVE','A_STD','D_MAX','D_AVE','P_N_030','P_N_3050','P_N_5080','P_N_80100','N_AVE']#,'P_D_12','P_D_23']
CYCLES_FEATURES = ['LABEL','N_MAX','N_AVE','A_MAX','A_AVE','A_STD','D_MAX','D_AVE','P_N_030','P_N_3050','P_N_5070','P_N_70100']#,'P_D_12','P_D_23']
CYCLES_OUTPUTS = 7

NN_TREND ="NN_DT"
TREND_FEATURES = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']
#TREND_OUTPUTS = '3'
TREND_OUTPUTS = 6

NN_EMOT = "NN_EMOT"
#EMOT_FEATURES = ['SPEED','POWER','SOC']
#EMOT_FEATURES = ['Tel','Vship','Tice','SOC']
EMOT_FEATURES = ['TQ_ICE','TQ_EMOT','TQ_LOAD','V_SHIP','N_ERROR','SOC']
EMOT_OUTPUTS = 1

NN_ENG = "NN_ICE"
#ENG_FEATURES = ['TREND','SPEED','POWER','SOC']
ENG_FEATURES = ['TQ_ICE','TQ_EMOT','TQ_LOAD','V_SHIP','N_ERROR','SOC']
ENG_OUTPUTS = 1

'''Common activation functions used for different Networks.

'''
LINEAR = "Linear"
TANH = "Tanh"
RELU = "Relu"
SIGMOID = "Sigmoid"
