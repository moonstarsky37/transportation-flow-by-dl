from params import *
import  numpy as np
from process import *
from ST_resnet import *
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.models import Model
#input shape
c_dim = (nb_channel, len_c,HEIGHT, WIDTH)
p_dim = (nb_channel, len_p,HEIGHT, WIDTH)
t_dim = (nb_channel, len_t,HEIGHT, WIDTH)


if __name__=='__main__':
    TrainX, TrainY, TestX, TestY = process_data()
    model = st_resnet(c_dim, p_dim, t_dim, residual_units=3, day_dim=-1)
    model.load_weights(filepath='./result/st_resnet.h5')
    model.summary()

    index=np.random.randint(TrainX[0].shape[0])
    c=TrainX[0][index]
    c=c[np.newaxis,:]
    p=TrainX[1][index]
    p=p[np.newaxis,:]
    t=TrainX[2][index]
    t=t[np.newaxis,:]
    X=[c,p,t]
    #{print(i,v) for i, v in enumerate(model.layers)}
    mid_model = Model(model.input,model.get_layer(index=3).output)
    output=mid_model.predict(X)
    output=output[0:]
    y=TrainY[index]
    np.save('./vision/label.npy',y)
    np.save('./vision/predict_conv_c_1.npy', output)
    # index
    f = open('./vision/index.txt', 'a')
    f.seek(0)
    f.truncate()  # 清空文件
    f.write("index: %d\n" %index)
    f.close()
