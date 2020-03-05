from keras.models import load_model
from Model import predict, generate_results, print_results_and_cm
import glob, os
import numpy as np

# loads the parameters saved after training the model. Ensures things like scalar and pre-processing type are the same
from save.parameters import pre_process_type, scalar, _size, batch_size
size = (_size, _size, 1)


save_folder = "save"
# loads the ZR and Keras model from file
model = load_model('{}/m.hdf5'.format(save_folder))
zr = x = np.load("{}/z.npy".format(save_folder), allow_pickle=True)[0] # [0] because the model is saved in an array

# path = "dataset_predict/"
path = "dataset_300/"

model_name = "custom"

input_size = 256

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

