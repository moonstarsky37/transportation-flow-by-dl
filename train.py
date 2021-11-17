from params import *
import  numpy as np
from process import *
from ST_resnet import *
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.models import Model
# data dim
c_dim = (nb_channel, len_c,HEIGHT, WIDTH)
p_dim = (nb_channel, len_p,HEIGHT, WIDTH)
t_dim = (nb_channel, len_t,HEIGHT, WIDTH)


def train(TrainX,TrainY):
    model = create_model()
    #train
    checkpointer = ModelCheckpoint(filepath='./result/st_resnet.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(TrainX,TrainY,batch_size=3,epochs=EPOCH,callbacks=[checkpointer,LR,early_stopping],validation_split=0.2)
    #validation
    keras_score = model.evaluate(TrainX, TrainY, verbose=1)
    rescaled_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO
    #print result
    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)

    return model

def testModel(model,testX, testY):

    model.load_weights('./result/st_resnet.h5')
    # model.load_weights('./result/original_st_resnet.h5')
    model.summary()

    XS, YS = testX, testY
    keras_score = model.evaluate(XS, YS, verbose=1)
    rescale_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    # 寫入
    f = open('./result/prediction_scores.txt', 'a')
    f.seek(0)
    f.truncate()  # 清空文件
    f.write("Keras MSE on TestData, %f\n" % keras_score)
    f.write("Rescaled MSE on TestData, %f\n" % rescale_MSE)
    f.close()
    pred = model.predict(XS, verbose=1, batch_size=BATCHSIZE)
    groundtruth = YS
    np.save('./result/STresnet_prediction.npy', pred)
    np.save('./result/STresnet_groundtruth.npy', groundtruth)
    
    
def create_model():
    model=st_resnet(c_dim,p_dim,t_dim,residual_units=3,day_dim=-1)
    model.summary()
    model.compile(loss='mse',optimizer='adam')
    return model
if __name__ == '__main__':
    # load data
    TrainX, TrainY, TestX, TestY = process_data()
    #train
    model= train(TrainX,TrainY)
    # model = create_model() # using trained model
    #test
    testModel(model,TestX, TestY)
    mse=0
    for i in range(TestY.shape[0]-1):
        front= TestY[i]
        last = TestY[i+1]
        mse += np.sum(np.sum((last-front)**2))
    mse =  mse/(TestY.shape[0]-1)
    rescaled_MSE = mse* MAX_FLOWIO * MAX_FLOWIO
    f = open('./result/prediction_scores.txt', 'a')
    f.write("Keras MSE on last_predict, %f\n" % mse)
    f.write("Rescaled MSE on last_predict, %f\n" % rescaled_MSE)
    f.close()
    print(mse)