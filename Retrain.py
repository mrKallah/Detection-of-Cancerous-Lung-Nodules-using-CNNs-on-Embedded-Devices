from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from Model import predict, generate_results, print_results_and_cm, train, train_zr
import glob, os
import numpy as np

# loads the parameters saved after training the model. Ensures things like scalar and pre-processing type are the same
from save.parameters import pre_process_type, scalar, _size, batch_size

size = (_size, _size, 1)

model_name = "custom"

# loads the ZR and Keras model from file
save_path = "save"

model = load_model('{}/m.hdf5'.format(save_path))

zr = x = np.load("{}/z.npy".format(save_path), allow_pickle=True)[0] # [0] because the model is saved in an array

input_size = 512

path = "dataset_retrain/"

# reads the folder
folder = glob.glob(os.path.join(path + "/*"))

# train and validate on 85% of the dataset (70% for train, 15% for val)
cut = int(np.asarray(folder).shape[0] * 0.85)
cut2 = int(np.asarray(folder).shape[0] * 0.15)

train_val = np.asarray(folder[:cut])

train_files = train_val[cut2:]
val_files = train_val[:cut2]

if batch_size > len(train_files):
    batch_size = len(train_files)

path = "dataset_predict/"
# reads the folder
folder = glob.glob(os.path.join(path + "/*"))
# train and validate on 85% of the dataset (70% for train, 15% for val)
tes = np.asarray(folder)


print("Predicting..")
model_acc, zr_acc, zr_y_preds, y_preds, y_tests = predict(model, zr, tes, pre_process_type, scalar, size, batch_size, input_size, model_name)
print("Predicting done")

print("Generating results")
zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm = generate_results(zr_acc, model_acc, y_tests, y_preds, zr_y_preds)
print("Results generated")

print_results_and_cm(zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm)

print("Training..")
# 0.8, 0.888, 0.444
model, history = train(model, train_files, val_files, pre_process_type, scalar, size, batch_size, output, epochs, model_name, represent, input_size)

zr = train_zr(zr, tes[:batch_size], pre_process_type, scalar, size, input_size)

print("Training done")

print("Predicting..")
model_acc, zr_acc, zr_y_preds, y_preds, y_tests = predict(model, zr, tes, pre_process_type, scalar, size, batch_size, input_size, model_name)
print("Predicting done")

print("Generating results")
zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm = generate_results(zr_acc, model_acc, y_tests, y_preds, zr_y_preds)
print("Results generated")

print_results_and_cm(zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm)

