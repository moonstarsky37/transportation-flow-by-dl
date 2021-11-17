import tensorflow as tf
import  keras
from keras.layers import Conv2D,Dropout,Input,BatchNormalization,Activation,add,Dense,Reshape
from keras.models import  Model
from  keras.engine.topology import Layer
from  keras import backend as K
import  numpy  as np
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

def st_resnet(c_dim=(2,3,19,18),p_dim=(2,1,19,18),t_dim=(2,1,19,18),residual_units=2,day_dim=-1):
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
        #conv1
        conv1=Conv2D(filters=64,kernel_size=(3,3),padding='same',data_format='channels_first')(input)
        #Residual Networks
        residual_output=res_units(filters=64,repetations=residual_units)(conv1)
        #conv2
        activation = Activation('relu')(residual_output)
        conv2 = Conv2D(filters=nb_channel, kernel_size=(3, 3), padding='same',data_format='channels_first')(activation)
        outputs.append(conv2)

    new_outputs=[]
    for output in outputs:
        new_outputs.append(iLayer()(output))
    main_outputs=add(new_outputs)
    if day_dim>0:
        day_input=Input(shape=(day_dim,))
        main_input.append(day_input)
        Dense1=Dense(10,activation='relu')(day_input)
        Dense2=Dense(map_height*map_width*nb_channel,activation='relu')(Dense1)
        day_output=Reshape((map_height,map_width,nb_channel))(Dense2)
        main_outputs=add([main_outputs,day_output])

    # main_outputs=Activation('relu')(main_outputs)
    main_outputs=Activation('tanh')(main_outputs)
    model=Model(inputs=main_input,outputs=main_outputs)
    return model

if __name__=='__main__':
    model=st_resnet()
    model.summary()

