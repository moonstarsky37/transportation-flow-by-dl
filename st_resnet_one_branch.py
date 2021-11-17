import tensorflow as tf
import  keras
from keras.layers import Conv2D,Dropout,Input,BatchNormalization,Activation,add,Dense,Reshape
from keras.models import  Model
from  keras.engine.topology import Layer
from  keras import backend as K
import  numpy  as np
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.models import Model
from  process import *
def residual_unit_(filters):
    def f(input):
        residual=BatchNormalization()(input)
        residual=Activation('relu')(residual)
        residual=Conv2D(filters,(3,3),padding='same',data_format='channels_first')(residual)
        residual=BatchNormalization()(residual)
        residual=Activation('relu')(residual)
        residual=Conv2D(filters,(3,3),padding='same',data_format='channels_first')(residual)
        return add([input,residual])
    return f

def res_units(filters, repetations):
    def f(input):
        for i in range(repetations):
            input = residual_unit_(filters)(input)
        return input
    return f

def st_resnet_one_branch(residual_units=2):
    '''
        Input
        C - Temperal closeness
        P - Period
        T - Trend
        dim = (map_height, map_width, nb_channel, len_sequence)
        number of residual units

        Return
        ST-ResNet model
    '''
    v=0
    input = Input(shape=(2*5,19,18))
    #conv1
    conv1=Conv2D(filters=64,kernel_size=(3,3),padding='same',data_format='channels_first')(input)
    #Residual Networks
    residual_output=res_units(filters=64,repetations=residual_units)(conv1)
    #conv2
    activation = Activation('relu')(residual_output)
    conv2 = Conv2D(filters=2, kernel_size=(3, 3), padding='same',data_format='channels_first')(activation)
    main_outputs=Activation('tanh')(conv2)
    model=Model(inputs=input,outputs=main_outputs)
    return model

if __name__=='__main__':
    model=st_resnet_one_branch(residual_units=2)
    model.summary()
    TrainX, TrainY, TestX, TestY = process_data()
    TrainX=np.concatenate((TrainX[0],TrainX[1],TrainX[2]),axis=1)
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    # train
    checkpointer = ModelCheckpoint(filepath='./result_one_branch/st_resnet.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(TrainX, TrainY, batch_size=100, epochs=100, callbacks=[checkpointer, LR, early_stopping],
              validation_split=0.2)
    # validation
    keras_score = model.evaluate(TrainX, TrainY, verbose=1)
    rescaled_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO
    # print result
    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)

