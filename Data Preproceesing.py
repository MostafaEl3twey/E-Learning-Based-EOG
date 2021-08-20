
import pandas as pd
import numpy as np
import csv
import scipy
from scipy import signal
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b, a] = butter(order, [low, high], btype='band')
    return b, a


# ------------------------------------------------------------------------------------------------------------------------------------------------
def ConCatenateChannels(T):
    #########################  Test ################################
    TestH = (pd.read_csv('/content/drive/MyDrive/Training/Trail ' + T + '/Horizontal ' + T + '/TestHRes ' + T + '.csv',
                         header=None)).to_numpy()
    DesiredTest = TestH[:, -1]
    DesiredTest = np.reshape(DesiredTest, (600, 1))
    TestH = TestH[:, :-1]

    TestV = (pd.read_csv('/content/drive/MyDrive/Training/Trail ' + T + '/Vertical ' + T + '/TestVRes ' + T + '.csv',
                         header=None)).to_numpy()
    TestV = TestV[:, :-1]
    Test = np.concatenate((TestH, TestV), axis=1)
    Test = np.concatenate((Test, DesiredTest), axis=1)
    with open(
            '/content/drive/MyDrive/Training/Trail ' + T + '/Output' + '/Data without Feature Ex' + '/Test ' + T + '.csv',
            'w', newline='\n') as file:
        writer = csv.writer(file)
        for row in Test:
            writer.writerow(row)
    #########################  Train  ################################
    TrainH = (
        pd.read_csv('/content/drive/MyDrive/Training/Trail ' + T + '/Horizontal ' + T + '/TrainHRes ' + T + '.csv',
                    header=None)).to_numpy()
    DesiredTrain = TrainH[:, -1]
    DesiredTrain = np.reshape(DesiredTrain, (2400, 1))
    TrainH = TrainH[:, :-1]

    TrainV = (pd.read_csv('/content/drive/MyDrive/Training/Trail ' + T + '/Vertical ' + T + '/TrainVRes ' + T + '.csv',
                          header=None)).to_numpy()
    TrainV = TrainV[:, :-1]
    Train = np.concatenate((TrainH, TrainV), axis=1)
    Train = np.concatenate((Train, DesiredTrain), axis=1)
    with open(
            '/content/drive/MyDrive/Training/Trail ' + T + '/Output' + '/Data without Feature Ex' + '/Train ' + T + '.csv',
            'w', newline='\n') as file:
        writer = csv.writer(file)
        for row in Train:
            writer.writerow(row)


# ------------------------------------------------------------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------------------------------------------------------------
def read_data(TargetChannel, fileType, TargetFileType, T):
    Data = pd.read_excel('/content/drive/MyDrive/Training/Trail ' + T + TargetChannel + T + TargetFileType + '.xlsx',
                         header=None, engine='openpyxl', )
    Data = Data.apply(pd.to_numeric, errors='coerce')
    Data = Data.iloc[:, :].values
    print("------------")
    return Data


# ------------------------------------------------------------------------------------------------------------------------------------------------

def preprocessing(T, Channels, Files, s=50):
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
            with open('/content/drive/MyDrive/Training/Trail ' + T + TargetChannel + T + ResampledFileName + T + '.csv',
                      'w', newline='\n') as file:
                writer = csv.writer(file)
                for Signal in ResampledData:
                    writer.writerow(Signal)


# ------------------------------------------------------------------------------------------------------------------------------------------------

Trial = 5
# Do Filter and resample on Data
preprocessing(str(Trial), Channels=2, Files=2, s=50)
ConCatenateChannels(str(Trial))
