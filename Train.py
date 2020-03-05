import numpy as np
import random
import os
from keras import backend as K
from Model import *


# disables tf warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# an effort to make TF deterministic on both cpu and gpu
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_DETERMINISTIC'] = '1'


import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from Model import __seed__
seed_value = __seed__

# set random seed in tensor flow
tf.set_random_seed(seed_value)

# set random seed in numpy
np.random.seed(seed_value)

# set random seed in random
random.seed(seed_value)

# set the sh seed for python
os.environ['PYTHONHASHSEED']=str(seed_value)

# create a new global tf session with the aim of increasing determinism of tf operations (not yet compatible with GPU)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from Utils import *
from zero_rule import ZeroRule
from Model import __version__

# todo: find optimal parameters
# todo: test training on more data
# todo: train model and save results
# todo: fix negative values displayed for training examples less than batch_size
# todo: find used ram / avaliable ram on XU4 NOTE: 2GB ram quad core 2GHz
# todo: test predicition time on XU4
# todo: update test predicting on XU4






print("Running v.{}".format(__version__))

# due to pre-processing steps, size should be no more than ca 70% of input image
# 512*0.7 = 358, 358%32 = 11, 32*11 = 352,
# 352 is the closest 32 divisible number to 70% of original size

represent = 0.6  # integer between 0 and 1, represents how much to over represent the underrepresented data
scalar = (-1, 1) # (-1, 1), (0, 1) and none
pre_process_type = "morphed" # None, "tozero", "resized", "morphed" or "max_morphed"
dropout = 0.02
conv_layer_1 = 32
_size = 252
epochs = 4
model_name = "custom" #"vgg16", "mobilenet", "resnet50", "custom"

# size of input images
input_size = 512


conv_layer_shape = (3, 3)
batch_size = 4

output = 8

if model_name == "custom":
    size = (_size, _size, 1)
else:
    size = (_size, _size, 3)

path = "dataset/"
conv_layer_2 = conv_layer_1
conv_layer_3 = conv_layer_2
conv_layer_4 = conv_layer_3
dense = conv_layer_3

history = None
x_data = np.asarray([])
y_data = np.asarray([])
x_error = np.asarray([])
y_error = np.asarray([])
i = 1

folder = glob.glob(os.path.join(path + "/*"))

# train and validate on 85% of the dataset (70% for train, 15% for val)
cut = int(np.asarray(folder).shape[0] * 0.85)
cut2 = int(np.asarray(folder).shape[0] * 0.15)

train_val = np.asarray(folder[:cut])
tes = np.asarray(folder[cut:])

train_files = train_val[cut2:]
val_files = train_val[:cut2]

if __name__=='__main__':

    zr = ZeroRule()

    print("Creating model")
    model = create_model(conv_layer_shape, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, size, dense, dropout, output, model_name)
    print("Model created")

    # print(model.summary())
    # exit()
    # plot_model(model, to_file='model_images/{}.png'.format(model_name))
    # exit()

    print("Training..")
    # 0.8, 0.888, 0.444
    model, history = train(model, train_files, val_files, pre_process_type, scalar, size, batch_size, output, epochs, model_name, represent, input_size)

    zr = train_zr(zr, train_files[:batch_size*100], pre_process_type, scalar, size, input_size)

    print("Training done")

    print("Predicting..")
    model_acc, zr_acc, zr_y_preds, y_preds, y_tests = predict(model, zr, tes, pre_process_type, scalar, size, batch_size, input_size, model_name)
    print("Predicting done")

    print("Generating results")
    zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm = generate_results(zr_acc, model_acc, y_tests, y_preds, zr_y_preds)
    print("Results generated")

    print_parameters(dropout, pre_process_type, conv_layer_1, conv_layer_shape, _size, scalar, timer, represent, epochs, model_name)
    print_results_and_cm(zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm)

    save_path = "save/"
    save_parameters(pre_process_type, scalar, _size, batch_size, save_path, output, epochs, input_size, represent, model, zr, history)
