import numpy as np
import matplotlib.pyplot as plt
import os,glob
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils.vis_utils import plot_model


def verbose_print(str="", verbose=True):
    """
    prints only if verbose is true, useful to ignore printing based on one value
    :param str: the string to be printed
    :param verbose: whether to print or not
    """
    if verbose:
        print(str)


def plot_preprocessing_images(imgs, title):
    """
    plots a 2x3 image plot
    :param imgs: array of images to plot
    :param title: the title of the plot
    """

    columns = 2
    rows = 4

    # create figure
    fig = plt.figure()
    # turn off the box around the image
    plt.box(False)
    # sets the title of the plot
    plt.title(title, {'fontweight':'bold','fontsize':11}, pad=10)

    plt.axis("off")
    plt.tight_layout()
    for i in range(1, columns * rows + 1):
        # adds a subplot
        fig.add_subplot(rows, columns, i)

        # turns off the axis and box and ensures padding
        plt.axis("off")
        plt.tight_layout(pad=10, w_pad=10, h_pad=1.0)

        # sets the title based on where in the loop we are
        if i == 1:
            title = "original"
        elif i == 2:
            title = "threshold"
        elif i == 3:
            title = "flood fill"
        elif i == 4:
            title = "crop"
        elif i == 5:
            title = "tozero"
        elif i == 6:
            title = "morph"
        elif i == 6:
            title = "max_morph"
        else:
            title = ""

        # sets the subplot title
        plt.title(title, {'fontsize':8})
        # plots the image onto the subplot
        if i < 8:
            plt.imshow(imgs[i-1], cmap=plt.cm.bone)

    # shows the image
    plt.show()


def plot_image(pixel_array, title):
    """
    plots one image
    :param pixel_array: the image to plot
    :param title: the title of the image
    :return:
    """

    # reshapes the iamge if needed
    try:
        if pixel_array.shape[2] == 1:
             pixel_array =  np.reshape(pixel_array, (pixel_array.shape[0], pixel_array.shape[1]))
    except:
        None

    #plots the image
    plt.imshow(pixel_array, cmap=plt.cm.bone)
    plt.xlabel('Height', fontsize=18)
    plt.ylabel('Width', fontsize=16)

    # changes the title to strings based the integers
    if title == 0:
        title = "No Nodule Present"
    elif title == 1:
        title = "Large Nodule"
    elif title == 2:
        title = "Small Nodule"
    elif title == 3:
        title = "Non-Nodule"
    plt.title(title)
    plt.show()


save_iterator = 0
def save_plot(pixel_array, title):
    """
    Does the same as show_file except it saves the plot to file instead of showing it
    :param pixel_array: the image
    :param title: the title
    """
    global save_iterator
    plt.imshow(pixel_array, cmap=plt.cm.bone)
    if title == 0:
        title = "No Nodule Present"
    elif title == 1:
        title = "Large Nodule"
    elif title == 2:
        title = "Small Nodule"
    elif title == 3:
        title = "Non-Nodule"
    plt.title(title)
    plt.savefig('output/{}.png'.format(save_iterator))
    save_iterator += 1

def get_files_in_folder(path, files=None):
    """
    searches for files in sub directories of input folder
    :param path: folder to search for folders in
    :param files: the array to append the files to
    :return:
    """

    # fixes a problem of files not being reset as files=[] is a mutable,
    # aka default, action and is ignored if files contains something
    if files == None:
        files = []

    for i in glob.glob(os.path.join(path + "*")):
        if os.path.isfile(i):
            files.append(path + os.path.basename(i))
        elif os.path.isdir(i):
            dirname = os.path.basename(i)
            get_files_in_folder(path + dirname + "/", files)
    return(files)


def normalize(min, max, x):
    """
    normallizes the x basedd on min and max
    :param min: the minimum observed value
    :param max: the maximum observed value
    :param x: the value to normalize
    :return: the normalized value
    """
    return (x - min) / (max - min)

def one_hot_encode(data):
    """
    one-hot encodes the input data
    :param data: the data to one-hot encode
    :return: a encoded version of the input array
    """

    # create a label encoder
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)

    # fit the label encoder to known data values
    unique_values = [0, 11, 12, 13, 14, 15, 20, 30]
    label_encoder = label_encoder.fit(unique_values)

    # encode dataset
    integer_encoded = label_encoder.transform(data)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # fit one hot encoder to known labels
    unique_values = np.asarray([[0], [1], [2], [3], [4], [5], [6], [7]])
    onehot_encoder = onehot_encoder.fit(unique_values)

    # encode the label encoded dataset as onehot encoeded
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    return onehot_encoded

def save_parameters(pre_process_type, scalar, _size, batch_size, path, output, epochs, input_size, represent, model, zr, history):
    '''
    Saves the parameters zr model, keras model and history to a file
    '''
    if not os.path.exists('{}'.format(path)):
        os.makedirs('{}'.format(path))

    if os.path.exists("{}/parameters.py".format(path)):
        os.remove("{}/parameters.py".format(path))

    os.system("echo pre_process_type = \\'{}\\' >> {}parameters.py".format(pre_process_type, path))
    os.system("echo scalar = \'{}\' >> '{}'parameters.py".format(scalar, path))
    os.system("echo _size = \'{}\' >> '{}'parameters.py".format(_size, path))
    os.system("echo batch_size = {} >> '{}'parameters.py".format(batch_size, path))


    os.system('echo output = {} >> "{}"parameters.py'.format(output, path))
    os.system('echo epochs = {} >> "{}"parameters.py'.format(epochs, path))
    os.system('echo input_size = {} >> "{}"parameters.py'.format(input_size, path))
    os.system('echo represent = {} >> "{}"parameters.py'.format(represent, path))


    model.save("{}m.hdf5".format(path))
    np.save('{}z.npy'.format(path), [zr])

    np.save('{}history.npy'.format(path), [history])
