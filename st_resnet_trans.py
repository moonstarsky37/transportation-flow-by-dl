import tensorflow as tf
import  keras
from keras.layers import Conv2D,Dropout,Input,BatchNormalization,Activation,add,Dense,Reshape,Conv2DTranspose
from keras.models import  Model
from  keras.engine.topology import Layer
from  keras import backend as K
import  numpy  as np
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.models import Model
from  process import *

#shape
c_dim = (nb_channel, len_c,HEIGHT, WIDTH)
p_dim = (nb_channel, len_p,HEIGHT, WIDTH)
t_dim = (nb_channel, len_t,HEIGHT, WIDTH)
class iLayer(Layer):
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])#输出0位置后的元素
        self.W = K.variable(initial_weight_value)
        self.trainable_weight = [self.W]
        self.built = True

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape

def output_unit_(filters):
    def f(input):
        cov=Conv2D(filters,(3,3),padding='valid',data_format='channels_first')(input)
        cov=BatchNormalization()(cov)
        cov=Activation('relu')(cov)
        cov=Conv2DTranspose(2,(3,3),padding='valid',data_format='channels_first')(cov)
        cov=BatchNormalization()(cov)
        cov=Activation('relu')(cov)
        return cov
    return f
def conv_unit_(filters):
    def f(input):
        cov=Conv2D(filters,(3,3),padding='valid',data_format='channels_first')(input)
        cov=BatchNormalization()(cov)
        cov=Activation('relu')(cov)
        cov=Conv2DTranspose(filters,(3,3),padding='valid',data_format='channels_first')(cov)
        cov=BatchNormalization()(cov)
        cov=Activation('relu')(cov)
        return cov
    return f

def conv_units(filters, repetations):
    def f(input):
        for i in range(repetations):
            input = conv_unit_(filters)(input)
        return input
    return f

def st_resnet_trans(c_dim=(2,3,19,18),p_dim=(2,1,19,18),t_dim=(2,1,19,18),residual_units=2):
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
    main_input=[]
    outputs=[]
    name=['c','p','t']
    v=0
    for input_size in [c_dim,p_dim,t_dim]:
        nb_channel, len_sequence,map_height, map_width=input_size
        input = Input(shape=(nb_channel*len_sequence,map_height,map_width))
        main_input.append(input)
        #Residual Networks
        cov_output=conv_units(filters=64,repetations=residual_units)(input)
        #conv2
        cov_output=output_unit_(filters=16)(cov_output)
        outputs.append(cov_output)

    new_outputs=[]
    for output in outputs:
        new_outputs.append(iLayer()(output))
    main_outputs=add(new_outputs)

    main_outputs=Activation('relu')(main_outputs)
    model=Model(inputs=main_input,outputs=main_outputs)
    return model

if __name__=='__main__':
    model=st_resnet_trans(c_dim,p_dim,t_dim,residual_units=1)
    model.summary()
    TrainX, TrainY, TestX, TestY = process_data()
    #model.summary()
    model.compile(loss='mse', optimizer='adam')
    # train
    checkpointer = ModelCheckpoint(filepath='./result_trans/st_resnet.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(TrainX, TrainY, batch_size=50, epochs=100, callbacks=[checkpointer, LR, early_stopping],
              validation_split=0.2)
    # validation
    keras_score = model.evaluate(TrainX, TrainY, verbose=1)
    rescaled_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO
    # print result
    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)

