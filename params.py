# INTERVAL = '30min'
TIMESTEP = 6
STARTDATE = '20161101'
ENDDATE = '20161130'
CITY = 'ChengDu'
INTERVAL = 2  # 60min
DAYTIMESTEP = int(24 * 60 / INTERVAL)+1
WEEKTIMESTEP = DAYTIMESTEP * 7 #every week
HEIGHT = 19
WIDTH = 18
CHANNEL = 2
BATCHSIZE = 100
nb_channel = 2
SPLIT = 0.2
LEARN = 0.0001
EPOCH = 5
LOSS = 'mse'
dayInfo=False
OPTIMIZER = 'adam'
len_c, len_p, len_t = 3, 1, 1
MAX_FLOWIO = 45.0
trainRatio = 0.8 # 80% training data
#dataPath = '../../{}/'.format(CITY)
#dataFile = dataPath + 'flowioK_{}_{}_{}_{}min.npy'.format(CITY, STARTDATE, ENDDATE, INTERVAL)