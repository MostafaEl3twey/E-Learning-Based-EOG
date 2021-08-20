from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


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
    y_test = YTest
    YTest = YTest - 1
    YTest = to_categorical(YTest)

    return XTrain, YTrain, XTest, YTest, y_test


def evaluate_model(trainX, trainy, testX, testy):
    n_outputs = trainy.shape[1]
    n_features = trainX.shape[1]
    trainX = np.expand_dims(trainX, axis=2)
    testX = np.expand_dims(testX, axis=2)
    verbose, epochs, batch_size = 0, 150, 128
    model = Sequential()
    model.add(
        Conv1D(filters=32, kernel_size=5, activation='relu', strides=1, padding='same', input_shape=(n_features, 1)))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', strides=1, padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))

    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    return accuracy, model


def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=1):
    Trial = 5
    # Do Filter and resample on Data
    # preprocessing(str(Trial),Channels=2, Files=2, s=100)
    # ConCatenateChannels(str(Trial))
    # Load Resampled Data File
    XTrain, YTrain, XTest, YTest, y_test = Load_ResampledData(str(Trial))
    # repeat experiment
    scores = list()
    Maxscore = 0
    acc = [0, 0, 0, 0, 0, 0]
    score, model = evaluate_model(XTrain, YTrain, XTest, YTest)
    XTest = np.expand_dims(XTest, axis=2)
    y_pred = np.argmax(model.predict(XTest), axis=-1)
    y_pred = np.reshape(y_pred, (len(y_pred), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    for i in range(y_pred.shape[0]):
        if y_pred[i][0] == 0 and y_test[i][0] - 1 == 0:
            acc[0] = acc[0] + 1
        elif y_pred[i][0] == 1 and y_test[i][0] - 1 == 1:
            acc[1] = acc[1] + 1
        elif y_pred[i][0] == 2 and y_test[i][0] - 1 == 2:
            acc[2] = acc[2] + 1
        elif y_pred[i][0] == 3 and y_test[i][0] - 1 == 3:
            acc[3] = acc[3] + 1
        elif y_pred[i][0] == 4 and y_test[i][0] - 1 == 4:
            acc[4] = acc[4] + 1
        elif y_pred[i][0] == 5 and y_test[i][0] - 1 == 5:
            acc[5] = acc[5] + 1
    score = score * 100.0
    # model.save("model1T2.h5")
    scores.append(score)
    # summarize results
    print(acc)
    summarize_results(scores)


# run the experiment
run_experiment()
