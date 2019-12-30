'''Tags used for Networks loading and creation.

'''
CREATE = 'create'
LOAD = 'load'
CLASSIFIERS = 'models'
CONTROLLERS = 'controllers'
NNCONTROLLER = 'NNCON'

'''Common tags used for graphs/figures and inference in Networks and saving in models.

'''
TRAIN = 'Train'
TEST = 'Test'
INFERENCE = 'Inference'
BUILD = 'build'
SIMULATION = 'simulation'
PREDICTION_LABEL = 'PRED_LABEL'

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
NN_CYCLES = 'NN_PAC'
CYCLES_FEATURES = ['LABEL','N_MAX','N_AVE','A_MAX','A_AVE','A_STD','D_MAX','D_AVE','P_N_030','P_N_3050','P_N_5070','P_N_70100','P_D_1','P_D_2','P_A_1','P_A_2']
CYCLES_FEATURES_INF = ['N_MAX','N_AVE','A_MAX','A_AVE','A_STD','D_MAX','D_AVE','P_N_030','P_N_3050','P_N_5070','P_N_70100','P_D_1','P_D_2','P_A_1','P_A_2']
CYCLES_OUTPUTS = 5

NN_TREND ='NN_CRT'
TREND_FEATURES = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']
TREND_FEATURES_INF = ['N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']

TREND_OUTPUTS = 6

NN_EMOT = 'NN_MOT'
EMOT_FEATURES = ['TQ_LOAD','V_SHIP','N_ERROR','SOC','DEAD_STOP','LOW_SPEED','MID_SPEED','HIGH_SPEED','ACCE','DECE']
EMOT_OUTPUTS = 1

NN_ENG = 'NN_ENG'
#ENG_FEATURES = ['TQ_LOAD','V_SHIP','N_ERROR','SOC','DEAD_STOP','LOW_SPEED','MID_SPEED','HIGH_SPEED','ACCE','DECE']
ENG_FEATURES = ['SE_REF','SE','SOC','LOAD_OBS','T_EL','T_ENG']
#ENG_FEATURES = ['SE','SOC','LOAD_OBS','T_EL','']

ENG_OUTPUTS = 1

'''Common activation functions used for different Networks.

'''
LAYER = 'Layer'
LINEAR = 'Linear'
TANH = 'Tanh'
RELU = 'Relu'
SIGMOID = 'Sigmoid'
SOFTMAX = 'Softmax'
FLATTEN = 'Flatten'

'''Common loss functions used for different Networks.

'''
SOFTMAX_CROSS_ENTROPY = 'softmax_cross_entropy'
MEAN_SQR_ERROR = 'mean_squared_error'
MEAN_ABS_ERROR = 'mean_absolute_error'
REDUCED_MEAN_SQR_ERROR = 'reduce_mean_square' 

'''Common metrics used for different Networks.

'''

TRAIN_LOSS = 'TRAIN_LOSS'
TEST_LOSS = 'TEST_LOSS'
TRAIN_ACC = 'TRAIN_ACC'
TEST_ACC = 'TEST_ACC'