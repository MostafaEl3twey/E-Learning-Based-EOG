from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
import csv
import scipy
from scipy import signal
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter
import time


# /content/drive/MyDrive/Trails/Trail 5

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b, a] = butter(order, [low, high], btype='band')
    return b, a


def ConCatenateChannels(T):
    #########################  Test ################################
    TestH = (pd.read_csv('/content/drive/MyDrive/Trails/Trail ' + T + '/Horizontal ' + T + '/TestHRes ' + T + '.csv',
                         header=None)).to_numpy()
    DesiredTest = TestH[:, -1]
    DesiredTest = np.reshape(DesiredTest, (600, 1))
    TestH = TestH[:, :-1]

    TestV = (pd.read_csv('/content/drive/MyDrive/Trails/Trail ' + T + '/Vertical ' + T + '/TestVRes ' + T + '.csv',
                         header=None)).to_numpy()
    TestV = TestV[:, :-1]
    Test = np.concatenate((TestH, TestV), axis=1)
    Test = np.concatenate((Test, DesiredTest), axis=1)
    with open('/content/drive/MyDrive/Trails/Trail ' + T + '/Output' + '/Test ' + T + '.csv', 'w',
              newline='\n') as file:
        writer = csv.writer(file)
        for row in Test:
            writer.writerow(row)
    #########################  Train  ################################
    TrainH = (pd.read_csv('/content/drive/MyDrive/Trails/Trail ' + T + '/Horizontal ' + T + '/TrainHRes ' + T + '.csv',
                          header=None)).to_numpy()
    DesiredTrain = TrainH[:, -1]
    DesiredTrain = np.reshape(DesiredTrain, (2400, 1))
    TrainH = TrainH[:, :-1]

    TrainV = (pd.read_csv('/content/drive/MyDrive/Trails/Trail ' + T + '/Vertical ' + T + '/TrainVRes ' + T + '.csv',
                          header=None)).to_numpy()
    TrainV = TrainV[:, :-1]
    Train = np.concatenate((TrainH, TrainV), axis=1)
    Train = np.concatenate((Train, DesiredTrain), axis=1)
    with open(
            '/content/drive/MyDrive/Trails/Trail ' + T + '/Output' + '/Data without Feature Ex' + '/Train ' + T + '.csv',
            'w', newline='\n') as file:
        writer = csv.writer(file)
        for row in Train:
            writer.writerow(row)


def SelectChannelAndFile(channel, FileType, T):
    if (channel == 0 and FileType == 0):  # Horizontal And Train
        TargetChannel = '/Horizontal '
        ResampledFileName = '/TrainHRes '
        TargetFileType = '/Train_H'
    elif (channel == 0 and FileType == 1):  # Horizontal And Test
        TargetChannel = '/Horizontal '
        ResampledFileName = '/TestHRes '
        TargetFileType = '/Test_H'
    elif (channel == 1 and FileType == 0):  # Vertical And Train
        TargetChannel = '/Vertical '
        ResampledFileName = '/TrainVRes '
        TargetFileType = '/Train_V'
    elif (channel == 1 and FileType == 1):  # Vertical And Test
        TargetChannel = '/Vertical '
        ResampledFileName = '/TestVRes '
        TargetFileType = '/Test_V'
    DataFile = read_data(TargetChannel, FileType, TargetFileType, T)
    return TargetChannel, DataFile, ResampledFileName


def read_data(TargetChannel, fileType, TargetFileType, T):
    Data = pd.read_excel('/content/drive/MyDrive/Trails/Trail ' + T + TargetChannel + T + TargetFileType + '.xlsx',
                         header=None, engine='openpyxl', )
    Data = Data.apply(pd.to_numeric, errors='coerce')
    # Data = Data.fillna(Data.mean())

    Data = Data.iloc[:, :].values
    print("------------")
    # print(Data)
    # Data=Data.to_numpy

    return Data


def preprocessing(T, Channels, Files, s=100):
    b, a = butter_bandpass(lowcut=0.5, highcut=20, fs=176, order=2)
    # Loop in two Channels
    for channel in range(Channels):
        # Loop in two Files(Train / Test)
        for FileType in range(Files):
            TargetChannel, DataFile, ResampledFileName = SelectChannelAndFile(channel, FileType, T)
            ResampledData = list()
            DesiredList = list()
            for Signal in DataFile:
                # print(Signal)
                nan_Sample = np.isnan(Signal)
                not_nan_sample = ~ nan_Sample
                Signal = Signal[not_nan_sample]
                # print(Signal)
                Target = Signal[-1]
                DesiredList.append(Target)
                Signal = np.delete(Signal, -1)

                # Filter And Resample The One Signal
                SignalAfterFilter = signal.lfilter(b, a, Signal)
                SignalAfterResample = scipy.signal.resample(SignalAfterFilter, s)
                ResampledData.append(SignalAfterResample)

            # Concate Signals With its Desired
            DesiredList = np.reshape(DesiredList, (-1, 1))

            ResampledData = np.concatenate((ResampledData, DesiredList), axis=1)
            with open('/content/drive/MyDrive/Trails/Trail ' + T + TargetChannel + T + ResampledFileName + T + '.csv',
                      'w', newline='\n') as file:
                writer = csv.writer(file)
                for Signal in ResampledData:
                    writer.writerow(Signal)


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
    y_test = YTest
    YTest = YTest - 1
    YTest = to_categorical(YTest)

    return XTrain, YTrain, XTest, YTest, y_test


def evaluate_model(trainX, trainy, testX, testy):
    n_outputs = trainy.shape[1]
    n_features = trainX.shape[1]
    trainX = np.expand_dims(trainX, axis=2)
    testX = np.expand_dims(testX, axis=2)
    verbose, epochs, batch_size = 0, 1000, 128

    model = Sequential()
    # 1
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation='relu', input_shape=(n_features, 1)))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 2
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 3
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 4
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 5
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    # fit network
    start = time.time()
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    end = time.time()
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, model, start, end


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
    for r in range(repeats):
        score, model, start, end = evaluate_model(XTrain, YTrain, XTest, YTest)
        XTest = np.expand_dims(XTest, axis=2)
        y_pred = np.argmax(model.predict(XTest), axis=-1)
        y_pred = np.reshape(y_pred, (len(y_pred), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        print(y_pred.shape)
        print(np.unique(y_pred, return_counts=True))
        print(np.unique(y_test, return_counts=True))
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
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
        print(f"Runtime of the program is {end - start}")
    # summarize results
    print(acc)
    summarize_results(scores)


# run the experiment
run_experiment()