import random
from keras import backend as K

from Model import *


import time

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

#create a new global tf session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from Utils import *
from zero_rule import ZeroRule

from Model import __version__
print("Running v.{}".format(__version__))


def train_en_mass(dropout, pre_process_type, conv_layer_1, conv_layer_shape, _size, scalar, path, represent, epochs, training_batch_size, batch_size, model_name):
    """
    Method used for gridsearch training
    """

    startTime = time.time()
    output = 8

    if model_name == "custom":
        size = (_size, _size, 1)
    else:
        size = (_size, _size, 3)

    conv_layer_2 = conv_layer_1
    conv_layer_3 = conv_layer_2
    conv_layer_4 = conv_layer_3
    dense = conv_layer_3

    # Create model
    model = create_model(conv_layer_shape, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, size, dense, dropout,
                         output, model_name)

    folder = glob.glob(os.path.join(path + "/*"))

    # train and validate on 85% of the dataset (70% for train, 15% for val)
    cut = int(np.asarray(folder).shape[0] * 0.85)
    cut2 = int(np.asarray(folder).shape[0] * 0.15)

    train_val = np.asarray(folder[:cut])
    tes = np.asarray(folder[cut:])

    train_files = train_val[cut2:]
    val_files = train_val[:cut2]
    # this is later used to take remove the validation from the train_validation object

    zr = ZeroRule()

    model, history = train(model, train_files, val_files, pre_process_type, scalar, size, batch_size, output, epochs, model_name, represent, input_size)

    zr = train_zr(zr, train_files[:batch_size*100], pre_process_type, scalar, size, input_size)

    print("Training done")

    print("Predicting..")
    model_acc, zr_acc, zr_y_preds, y_preds, y_tests = predict(model, zr, tes, pre_process_type, scalar, size, batch_size, input_size, model_name)
    print("Predicting done")

    print("Generating results")
    zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm = generate_results(zr_acc, model_acc, y_tests, y_preds, zr_y_preds)

    timer = time.time() - startTime

    return zr_acc, model_acc, b_zr_acc, b_model_acc, timer, zr, model, history

import warnings
warnings.filterwarnings("ignore")

output_file = "output7.txt"
if __name__ == "__main__":
    input_size = 512

    represent = [0.3]
    scalar = [(0, 1)]
    pre_process_type = ["morphed"]
    dropout = [0.02]
    conv_layer_1 = [64]
    _size = [128]
    epochs = [16]
    model_name = ["resnet50", "mobilenet", "vgg16"]




    #####################
    training_batch_size = 0
    batch_size = 5
    should_save = True


    path = "dataset/"

    total = len(dropout) * len(pre_process_type) * len(conv_layer_1) * len(_size) * len(scalar) * len(represent) * len(epochs) * len(model_name)


    test = None
    for s in range(len(_size)):
        for c in range(len(conv_layer_1)):
            for d in range(len(dropout)):
                for p in range(len(pre_process_type)):
                    for sc in range(len(scalar)):
                        for r in range(len(represent)):
                            for e in range(len(epochs)):
                                for m in range(len(model_name)):
                                    if test is None:
                                        test = [[s, c, d, p, sc, r, e, m]]
                                    else:
                                        test.append([s, c, d, p, sc, r, e, m])

    i = 0
    if i == 0:
        f = open(output_file, "w+")
        f.write("dropout\tpre_processing_type\tconv_layer_1\tconv_layer_shape\tsize\tscalar\trepresent\tepochs\tmodel\ttime\tmulti_class_zr_acc\tmulti_class_model_acc\tboolean_zr_acc\tboolean_model_acc\trun number\n")
        f.close()
    for i in range(i, int(total)):
        s = _size[test[i][0]]
        c = conv_layer_1[test[i][1]]
        d = dropout[test[i][2]]
        p = pre_process_type[test[i][3]]
        sc = scalar[test[i][4]]
        r = represent[test[i][5]]
        e = epochs[test[i][6]]
        m = model_name[test[i][7]]

        print("Run {}/{}, {}% done".format(i+1, total, (i/total)*100))
        print_parameters(d, p, c, (3, 3), s, sc, 0, r, e, m)
        zr_acc, model_acc, b_zr_acc, b_model_acc, timer, zr, model, history = train_en_mass(d, p, c, (3, 3), s, sc, path, r, e, training_batch_size, batch_size, m)
        print_parameters(d, p, c, (3, 3), s, sc, timer, r, e, m)
        print_results(zr_acc, model_acc, b_zr_acc, b_model_acc)
        write_results(zr_acc, model_acc, b_zr_acc, b_model_acc, timer, d, p, c, (3, 3), s, sc, r, i, output_file, e, m)

        if should_save:
            save_path = "save_{}/".format(i+1)
            save_parameters(p, sc, s, batch_size, save_path, 8, e, input_size, r, model, zr, history)

        model_reset(model, zr)


        i += 1
