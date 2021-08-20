from numpy import mean
from numpy import std
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.pooling import AveragePooling1D, GlobalAveragePooling1D
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.utils import to_categorical



# /content/drive/MyDrive/Trails/Trail 5


def Load_ResampledData(T):
    TrainData = (pd.read_csv(
        '/content/drive/MyDrive/Trails/Trail ' + T + '/Output' + '/Data without Feature Ex' + '/Train ' + T + '.csv',
        header=None)).to_numpy()

    XTrain = TrainData[:, :-1]
    YTrain = TrainData[:, -1]
    print(XTrain.shape, YTrain.shape)
    YTrain = YTrain - 1
    YTrain = to_categorical(YTrain)

    TestData = (pd.read_csv(
        '/content/drive/MyDrive/Trails/Trail ' + T + '/Output' + '/Data without Feature Ex' + '/Test ' + T + '.csv',
        header=None)).to_numpy()

    XTest = TestData[:, :-1]
    YTest = TestData[:, -1]
    print(XTest.shape, YTest.shape)
    YTest = YTest - 1
    YTest = to_categorical(YTest)

    return XTrain, YTrain, XTest, YTest


def stem(input):
    conv = Conv1D(64, 3, padding='same', strides=1, activation='relu')(input)

    pool = MaxPooling1D(3, strides=2, padding='same')(conv)

    conv = Conv1D(192, 3, strides=1, padding='same', activation='relu')(pool)
    pool = MaxPooling1D(3, strides=2, padding='same')(conv)

    return pool


def auxiliary(input, name=None):
    x = AveragePooling1D(pool_size=5, strides=3, padding="same")(input)
    x = Conv1D(filters=128, kernel_size=1, padding="same", strides=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dropout(0.7)(x)
    x = Dense(units=6, activation="softmax", kernel_initializer="he_normal", name=name)(x)
    return x


def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = Conv1D(f1, 1, padding='same', strides=1, activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv1D(f2_in, 1, padding='same', strides=1, activation='relu')(layer_in)
    conv3 = Conv1D(f2_out, 3, padding='same', strides=1, activation='relu')(conv3)
    # 5x5 conv
    conv5 = Conv1D(f3_in, 1, padding='same', strides=1, activation='relu')(layer_in)
    conv5 = Conv1D(f3_out, 5, padding='same', strides=1, activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPooling1D(3, strides=1, padding='same')(layer_in)
    pool = Conv1D(f4_out, 1, padding='same', strides=1, activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def evaluate_model(trainX, trainy, testX, testy):
    n_outputs = trainy.shape[1]
    n_features = trainX.shape[1]
    trainX = np.expand_dims(trainX, axis=2)
    testX = np.expand_dims(testX, axis=2)
    verbose, epochss, batch_size = 0, 500, 1024
    #################################################################
    # define model input
    visible = Input(shape=(n_features, 1))
    layer = stem(visible)

    # add inception module
    # -------------------------------------------------------------------
    # 3a
    layer = inception_module(layer, 64, 96, 128, 16, 32, 32)
    # 3b
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    layer = MaxPooling1D(3, strides=2, padding="same")(layer)
    # -------------------------------------------------------------------
    # 4a
    layer = inception_module(layer, 192, 96, 208, 16, 48, 64)
    # aux1 = auxiliary(layer, name='aux1')
    # -------------------------------------------------------------------
    # 4b
    layer = inception_module(layer, 160, 112, 224, 24, 64, 64)
    # 4c
    layer = inception_module(layer, 128, 128, 256, 24, 64, 64)
    # 4d
    layer = inception_module(layer, 112, 144, 288, 32, 64, 64)
    # aux2 = auxiliary(layer, name='aux2')
    # -------------------------------------------------------------------
    # 4e
    layer = inception_module(layer, 256, 160, 320, 32, 128, 128)
    layer = MaxPooling1D(3, strides=2, padding="same")(layer)
    # -------------------------------------------------------------------
    # 5a
    layer = inception_module(layer, 256, 160, 320, 32, 128, 128)
    # 5b
    layer = inception_module(layer, 384, 192, 384, 48, 128, 128)
    # ------------------------------------------------------------------------

    layer = GlobalAveragePooling1D()(layer)

    layer = Dropout(0.4)(layer)

    output = Dense(units=6, activation="softmax")(layer)

    #################################################################
    model = Model(inputs=visible, outputs=output)
    # plot_model(model, show_shapes=True, to_file='inception_module.png')

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    model.fit(trainX, trainy, epochs=2500, batch_size=64, verbose=1)

    _, accuracy = model.evaluate(testX, testy, verbose=0)

    return accuracy, model


def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=1):
    Trial = 1
    XTrain, YTrain, XTest, YTest = Load_ResampledData(str(Trial))
    # repeat experiment
    scores = list()
    Maxscore = 0
    for r in range(repeats):
        score, model = evaluate_model(XTrain, YTrain, XTest, YTest)
        score = score * 100.0
        model.save_weights("model1.h5")
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
run_experiment()