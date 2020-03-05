

import numpy as np
from random import seed

seed(0)






def represent_underepresented_classes(x_data, y_data, represent):
    """
    Restructures data to contain same amount of all samples
    :param x_data: data to be represented
    :param y_data: classes, must be integers. use encoding to ensure.
    :return:
    """



    # Sort the x and y data using merge sort
    ind = np.argsort(y_data, kind="mergesort")
    tmpx = []
    tmpy = []
    for i in range(len(ind)):
        tmpx.append(x_data[ind[i]])
        tmpy.append(y_data[ind[i]])
    x_data = np.asarray(tmpx)
    y_data = np.asarray(tmpy)
    del tmpx
    del tmpy

    # extract the classes and sort the data under specific classes
    classes = []
    data = [[]]
    for i in range(len(y_data)):
        if y_data[i] not in classes:
            classes.append(y_data[i])
            if data != [[]]:
                data.append([x_data[i]])
            else:
                data[0] = [x_data[i]]
        else:
            ind = np.where(classes==y_data[i])
            data[ind[0][0]].append(x_data[i])

    # find the most occurring class
    max = 0
    for i in range(len(classes)):
        if max < len(data[i]):
            max = len(data[i])

    # duplicate underrepresented classes
    for da in range(len(data)):
        if len(data[da]) != max:
            for i in range(int(max*represent) - len(data[da])):
                data[da].append(data[da][i])

    # create x and y data with of the same size as each other
    x_data = []
    y_data = []
    for c in range(len(classes)):
        for d in range(len(data[c])):
            x_data.append(data[c][d])
            y_data.append(classes[c])

    return np.asarray(x_data), np.asarray(y_data)

def get_loss_curve(history, name):
    '''
    Prints the values needed to create a loss curve for a history object.
    :param folder:
    :return:
    '''

    print("#####################")
    print(name)
    print(history.history['acc'])
    print(history.history['val_acc'])
    print(history.history['loss'])
    print(history.history['val_loss'])
    print("#####################")


if __name__=="__main__":
    test_type = "loss_curve"

    if test_type == "represent":
        represent = 0.3
        x_data = np.asarray(["five1", "six1", "seven1", "eight1", "nine1", "zero1", "one1", "two1", "three1", "four1", "nine2", "zero2", "one2", "two2", "one3", "one4", "one5", "one6", "one7", "one8", "zero3", "zero4", "zero5"])
        y_data = np.asarray([5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 9, 0, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        print(x_data.shape)
        print(y_data.shape)
        x_data, y_data = represent_underepresented_classes(x_data, y_data, represent)
        print(x_data)
        print(y_data)
    elif test_type == "loss_curve":
        for i in range(0,1):
            folder = "save"
            # folder = "save_{}".format(i)
            history = np.load("{}/history.npy".format(folder), allow_pickle=True)[0]  # [0] because the model is saved in an array
            get_loss_curve(history, folder)

