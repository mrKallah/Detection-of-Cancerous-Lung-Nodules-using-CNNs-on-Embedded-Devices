from process import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import reset_default_graph
from keras.backend import clear_session
import gc
import random

from keras.utils import Sequence
from Utils import one_hot_encode
from sklearn.utils import shuffle
from visuallize import represent_underepresented_classes
import numpy as np

__version__ = 1.9
__seed__ = 1

random.seed(__seed__)


def boolify(y):
	"""
	Takes an input of not one-hot encoded (0, 1, 2, 3.. etc) inputs and converts it to a boolean output.
	Example: y = [0, 0, 3, 0, 1, 0, 5, 7] -> f = [0, 0, 1, 0, 1, 0, 1, 1]
	:param y: Input of array of classes where 0 signifies one class and any other value signifies the other class
	:return: The converted output as array of same size as input array (type = list)
	"""
	f = []
	for x in y:
		if x == 0:
			f.append(0)
		else:
			f.append(1)
	return f


def read_file(file, x_data, y_data, pre_process_type, scale, size, input_size):
	"""
	Reads a file from array, pre-processes it, appends the x and y data to the the input values.
	:param file: File name of file to read
	:param x_data: a list to append the x data to
	:param y_data: a list to append the y data to
	:param pre_process_type: the type of pre-processing, can be "none", "tozero", "resized", "morphed" or "max_morphed"
	:param size: the wanted output size, the pre-processing steps, if not none,
					crops the image and thus the output size should be ~70% of the original size
					However, due to the parameters, the image size should be possible to do modulo 32 with
					so for an image of 512*512 the output should be int((512*0.7)/32)*32 = 352
	:return: the x and y data arrays with the newest file appended to the end.
	"""

	# read the Output class containing the pixel data, malignancy value and nodule class
	x = np.load(file, allow_pickle=True)
	img = x[0].pixel_array

	# uncomment this to ensure files read are in expected format
	# show_file(img, "test")
	# exit()

	img = img.reshape(input_size, input_size, 1)

	# preprocess and resize the image to chosen input size
	if (pre_process_type != None) & (pre_process_type != 'None'):
		img = preprocess(img, (size[0], size[1]), pre_process_type)
	else:
		img = cv2.resize(img, (size[0], size[1]))

	# Scale the values in the image to be between the values of the scale variable as long as scale != None
	# example of scale with (-1, 1) for arr = [0, 2, 0, 1, 2, 1] -> arr = [-1, 1, -1, 0, 1, 0]
	if scale != None:
		scaler = MinMaxScaler(feature_range=scale)
		scaler = scaler.fit(img)
		img = scaler.transform(img)

	# add data to array for later use
	x_data = np.append(x_data, img)

	# convert nodule class and malignancy into class and subclass by connecting together
	# if class is 5 and malignancy is 4 then this wil correspond to 54 and if class is
	# 3 and malignancy is 0 it will correspond to 30
	# this will later be used for one-hot encoding
	y_data = np.append(y_data, np.asarray(["{}{}".format(x[0].nodule_class, x[0].malignancy)], dtype=int))

	return x_data, y_data


def format_data(x_data, y_data, size):
	"""
	Reshapes, coverts to np array
	:param x_data: the expected input array for training
	:param y_data: the expected output array for training
	:param size: the output size
	:return: correctly shaped x and y data
	"""
	# make sure data is in nparray and one hot encode output
	x_data = np.asarray(x_data).reshape(-1, size[0], size[1], 1)
	y_data = np.asarray(y_data)

	x_data, y_data = shuffle(x_data, y_data)

	return x_data, y_data


def fit_on_data(x_train, y_train, x_val, y_val, model, epochs, training_batch_size, verbose=1):
	"""
	Trains the model and zr using the training and validation data.
	:param x_train: the expected input array for training
	:param y_train: the expected output array for training
	:param x_val: the expected input array for validation
	:param y_val: the expected input array for validation
	:param model: the keras model that will be used for training
	:param zr: the zr model which is used as a baseline (will only be trained once)
	:param epochs: the mount of times to train on the entire dataset
	:param training_batch_size: the amount of training examples per training cycle (bigger is better for acc but takes up more memory)
	:param verbose: whether or not to print the keras progression
	:return: the keras model, the zero rule model, wrongly predicted x and y values from the training set
	"""
	# train the model
	model.fit(x_train, y_train, validation_data=(np.asarray(x_val), np.asarray(y_val)), epochs=epochs,
			  batch_size=training_batch_size, verbose=verbose)

	# old implementation of represented underrepresented data
	# # predict the y values for the x training data
	# y_pred = model.predict(x_train)
	#
	# x_error = []
	# y_error = []
	#
	# # converts results from normalized to zeros and ones
	# new_pred = np.zeros(y_pred.shape)
	# for j in range(y_pred.shape[0]):
	#	 new_pred[j][np.argmax(y_pred[j])] = 1
	# y_pred = new_pred
	# del new_pred
	#
	# # saves any predictions that are wrong
	# for i in range(y_pred.shape[0]):
	#	 if not np.min(np.equal(y_pred[i], y_train[i])):
	#		 x_error.append(x_train[i])
	#		 y_error.append(y_train[i])

	return model


def __predict__(x_test, y_test, model, zr, y_preds, y_tests, zr_y_preds, size, zr_acc, model_acc, model_name):
	"""
	Uses the keras and zr models to predict based on the given features.
	:param x_test: the expected input array for testing
	:param y_test: the expected input array for testing
	:param model: the trained keras model
	:param zr: the trained zr model
	:param y_preds: an array to store the model predicted results in
	:param y_tests: an array to store the true values in
	:param zr_y_preds: an array to store the zr predicted results in
	:param size: the expected outptut size
	:param zr_acc: an array to store the zr accuracies
	:param model_acc: an array to store the model accuracies
	:return: the keras model, predicted results from model and zr along with the true results and lastly the accurarcies for zr and model
	"""

	x_test, y_test = format_data(x_test, y_test, size)

	# imagenet requires rgb images not greyscale
	if model_name != "custom":
		new = []
		for x in x_test:
			shape = (size[0], size[1])
			x = np.reshape(x, shape)
			x = np.asarray([x, x, x])
			shape = (size[0], size[1], 3)
			x = np.reshape(x, shape)
			new.append(x)

			x_test = np.asarray(new)

	y_test = one_hot_encode(y_test)

	# convert to nparray
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	# use the zr model to predict a baseline
	zr_y_pred = zr.predict(np.asarray(x_test))
	zr_acc.append(accuracy_score(np.asarray(y_test), zr_y_pred))

	# use the keras model to predict the actual results
	y_pred = model.predict(x_test)

	# append the results as due to memory constraints the prediction may be done in batches.
	new_pred = np.zeros(y_pred.shape)
	for j in range(y_pred.shape[0]):
		new_pred[j][np.argmax(y_pred[j])] = 1
	y_pred = new_pred
	del new_pred

	model_acc.append(accuracy_score(np.asarray(y_test), y_pred))

	# append the zr and model predicted and true values to arrays for confusion matrices
	zr_y_preds.append(zr_y_pred)
	y_preds.append(y_pred)
	y_tests.append(y_test)

	return model_acc, zr_acc, zr_y_preds, y_preds, y_tests


def convert_results(y_tests, y_preds, zr_y_preds):
	"""
	converts y values for true and predicted outputs from one hot encoded results to integer based reults, example [1, 0, 0, 0] -> 0 and [0, 0, 0, 1] -> 3
	:param y_tests: the true reults
	:param y_preds: the keras model predicted results
	:param zr_y_preds: the zero rule predicted results
	:return: return the converted inputs
	"""
	y = []
	z = []
	m = []

	# do multiple result conversions in one loop to save on for loops
	for j in range(len(y_tests)):
		for k in range(len(y_tests[j])):
			y.append(y_tests[j][k])
			z.append(zr_y_preds[j][k])
			m.append(y_preds[j][k])

	y = [np.where(r == 1)[0][0] for r in y]
	z = [np.where(r == 1)[0][0] for r in z]
	m = [np.where(r == 1)[0][0] for r in m]
	return y, z, m


def create_model(conv_layer_shape, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, size, dense, dropout, output,
				 model_name):
	'''
	creates the keras models
	:return: the created model
	'''

	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
	from keras.layers.normalization import BatchNormalization
	from keras.applications.mobilenet_v2 import MobileNetV2
	from keras.applications.vgg16 import VGG16
	from keras.applications.resnet50 import ResNet50
	from keras import Model

	if model_name == "custom":
		# Create model
		model = Sequential()

		model.add(Conv2D(conv_layer_1, conv_layer_shape, input_shape=size))
		model.add(BatchNormalization(axis=-1))
		model.add(Activation('relu'))

		model.add(Conv2D(conv_layer_2, conv_layer_shape))
		model.add(BatchNormalization(axis=-1))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(conv_layer_3, conv_layer_shape))
		model.add(BatchNormalization(axis=-1))
		model.add(Activation('relu'))

		model.add(Conv2D(conv_layer_4, conv_layer_shape))
		model.add(BatchNormalization(axis=-1))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())

		# Fully connected layer
		model.add(Dense(dense))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(dropout))
		model.add(Dense(output))

		model.add(Activation('sigmoid'))

	elif model_name == "mobilenet":
		base = MobileNetV2(input_shape=size, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None)
		out = Sequential()
		out.add(Flatten())
		out.add(Dense(output))
		model = Model(inputs=base.input, outputs=out(base.output))

	elif model_name == "vgg16":
		base = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=size, pooling=None)
		out = Sequential()
		out.add(Flatten())
		out.add(Dense(output))
		model = Model(inputs=base.input, outputs=out(base.output))

	elif model_name == "resnet50":
		base = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=size, pooling=None)
		out = Sequential()
		out.add(Flatten())
		out.add(Dense(output))
		model = Model(inputs=base.input, outputs=out(base.output))

	else:
		raise ValueError('unknown model name, got: {}'.format(model_name))

	model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
	return model


class DataGenerator(Sequence):
	"""
	DataGenerator class for reading the data for the Keras model, based on the included keras data generator
	"""

	def __init__(self, image_filenames, batch_size, size, pre_process_type, scale, input_size, represent, model_name):
		self.image_filenames = image_filenames
		self.size = size
		self.batch_size = batch_size
		self.pre_process_type = pre_process_type
		self.scale = scale
		self.input_size = input_size
		self.represent = represent
		self.model_name = model_name

	def __len__(self):
		return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

	def __getitem__(self, idx):
		'''
		reads files in small batches for the training method
		'''
		batch = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]

		x_data, y_data = np.asarray([]), np.asarray([])

		for file in batch:
			x_data, y_data = read_file(file, x_data, y_data, self.pre_process_type, self.scale, self.size,
									   self.input_size)

		x_data, y_data = format_data(x_data, y_data, self.size)

		# imagenet requires rgb images not greyscale
		if self.model_name != "custom":
			new = []
			for x in x_data:
				shape = (self.size[0], self.size[1])
				x = np.reshape(x, shape)
				x = np.asarray([x, x, x])
				shape = (self.size[0], self.size[1], 3)
				x = np.reshape(x, shape)
				new.append(x)

			x_data = np.asarray(new)

		x_data, y_data = represent_underepresented_classes(x_data, y_data, self.represent)
		x_data, y_data = shuffle(x_data, y_data)

		y_data = one_hot_encode(y_data)

		return x_data, y_data


def train(model, train, val, pre_process_type, scalar, size, batch_size, output, epochs, model_name, represent, input_size, verbose=1):
	'''
	Trains the model on the training data
	:param model: The model to be trained
	:param train: Training set
	:param val: Validation set
	:param pre_process_type: the type of preprocessing to apply to the images
	:param scalar: the type of scalar
	:param size: the image size as a tuple
	:param batch_size: the size of the training batches
	:param output: the amount of output classes, this is no longer used
	:param epochs: How many epochs to train for
	:param model_name: the name of the model training
	:param represent: how much to over represent the under represented classes
	:param input_size: the size of the images as files. Not to be confused with size which is the wanted training size
	:param verbose: whether to print or not
	:return:
	'''
	num_training_samples = len(train)
	num_validation_samples = len(val)

	training_batch_generator = DataGenerator(train, batch_size, size, pre_process_type, scalar, input_size, represent,
											 model_name)
	validation_batch_generator = DataGenerator(val, batch_size, size, pre_process_type, scalar, input_size, represent,
											   model_name)
	history = model.fit_generator(generator=training_batch_generator,
						steps_per_epoch=(num_training_samples // batch_size),
						epochs=epochs,
						validation_data=validation_batch_generator,
						validation_steps=(num_validation_samples // batch_size),
						verbose=verbose)

	return model, history


def train_zr(zr, train, pre_process_type, scale, size, input_size):
	'''
	Trains the ZR classifier.
	:param zr: the model to train
	:param train: the training data
	:param pre_process_type: the pre-processing type to apply (this should be none as the classifier does not look at the image data)
	:param scale: The scale of the images
	:param size: the size of the images
	:param input_size: the input size, I.E. the size of the images in file
	:return:
	'''
	x_data, y_data = np.asarray([]), np.asarray([])

	for file in train:
		x_data, y_data = read_file(file, x_data, y_data, pre_process_type, scale, size, input_size)

	x_data, y_data = format_data(x_data, y_data, size)

	y_data = one_hot_encode(y_data)

	zr.fit([], y_data)
	return zr


def train_old(model, zr, train, val, pre_process_type, scalar, size, batch_size, output, epochs, training_batch_size,
			  represent, input_size, verbose=1):
	"""
	Trains a model on the files in the train_val array
	:param model: the keras model to train
	:param zr: the zero rule model to train
	:param train: the list of files to train on (as strings, example: dataset/0.npy)
	:param val: the list of files to validate on (as strings, example: dataset/0.npy)
	:param pre_process_type: the type of pre-processing
	:param scalar: the scalar type
	:param size: the wanted output size
	:param batch_size: how many examples to read before training
	:param output: how many values are expected as outputs
	:param epochs: how many times to train on all the data
	:param training_batch_size: the amount of data to feed to the algorithm each time
	:param verbose: whether to enable keras progress or not
	:return: the keras and zr model and any missclassified examples after training
	"""
	x_train = np.asarray([])
	y_train = np.asarray([])
	x_val = np.asarray([])
	y_val = np.asarray([])
	x_data = np.asarray([])
	y_data = np.asarray([])
	i = 1
	j = 0

	# go through all files to read and train on them
	for file in train:

		x_train, y_train = read_file(file, x_train, y_train, pre_process_type, scalar, size, input_size)

		if (int(i) % int(batch_size * 0.7) == 0) | (int(i) >= int(len(train))):
			for _ in range(int(batch_size * 0.15)):
				if j > len(val):
					None;
				else:
					x_val, y_val = read_file(val[j - 1], x_val, y_val, pre_process_type, scalar, size, input_size)
				j += 1

			x_train, y_train = format_data(x_train, y_train, size)
			x_val, y_val = format_data(x_val, y_val, size)

			from visuallize import represent_underepresented_classes
			print("Representing underepresented classes")
			x_train, y_train = represent_underepresented_classes(x_train, y_train, represent)

			x_train, y_train = shuffle(x_train, y_train)

			y_train = one_hot_encode(y_train)
			y_val = one_hot_encode(y_val)

			# trains the zr model in the first run
			# note it trains on validation data as the training data is compensated for the underrepresented classes
			if not zr.trained:
				zr.fit(x_val, y_val)

			# old implementation of represented underrepresented data
			# if random.randint(0, 100) <= re_train_on_known_errors:
			#	 for error in range(x_error.shape[0]):
			#		 x_train = np.append(x_train, x_error[error])
			#		 y_train = np.append(y_train, y_error[error])
			#		 x_train = np.asarray(x_train).reshape(-1, size[0], size[1], size[2])
			#		 y_train = np.asarray(y_train).reshape(-1, output)

			min = i - int(batch_size * 0.7)
			if min < 0:
				min = 0
			print("training on {} to {} out of {}".format(min, i, len(train)))
			model = fit_on_data(x_train, y_train, x_val, y_val, model, epochs,
								training_batch_size, verbose=verbose)

			x_train = np.asarray([])
			y_train = np.asarray([])
			x_val = np.asarray([])
			y_val = np.asarray([])

			print("Reading files")
		i += 1
	return model, zr


def predict(model, zr, tes, pre_process_type, scalar, size, batch_size, input_size, model_name, verbose=0):
	"""
	Predicts outputs based on features in batches
	:param model: keras model
	:param zr: zero rule model
	:param tes: the list of files to test on (as strings, example: dataset/0.npy)
	:param pre_process_type: the type of pre-processing
	:param scalar: the type of scaling
	:param size: the expected output size
	:param batch_size: the amount of files to read between each time a prediction session is done
	:return: arrays with the accuracies, arrays with the predicted and true values
	"""
	x_test = np.asarray([])
	y_test = np.asarray([])
	zr_acc = []
	model_acc = []
	zr_y_preds = []
	y_preds = []
	y_tests = []
	i = 1
	for file in tes:
		x_test, y_test = read_file(file, x_test, y_test, pre_process_type, scalar, size, input_size)

		if (i % batch_size == 0) | (i == tes.shape[0] - 1):
			# x_data, y_data = format_data(x_data, y_data, size)

			# reset the ZR
			if verbose == 1:
				print("predicting on {} to {}".format(i - batch_size, i))
			model_acc, zr_acc, zr_y_preds, y_preds, y_tests = __predict__(x_test, y_test, model, zr, y_preds,
																		  y_tests, zr_y_preds, size, zr_acc,
																		  model_acc, model_name)
			if verbose == 1:
				print("Reading files")
			x_test = []
			y_test = []
		i += 1

	return model_acc, zr_acc, zr_y_preds, y_preds, y_tests


def print_parameters(dropout, pre_process_type, conv_layer_1, conv_layer_shape, _size, scalar, timer,
					 re_train_on_known_errors, epochs, model):
	"""
	Prints the paremeters and time taken training
	"""
	print("#######  Parameters  #######")
	print("dropout = {}".format(dropout))
	print("pre_process_type = {}".format(pre_process_type))
	print("conv_layer_1 = {}".format(conv_layer_1))
	print("conv_layer_shape = {}".format(conv_layer_shape))
	print("_size = {}".format(_size))
	print("scalar = {}".format(scalar))
	print("time = {}".format(timer))
	print("re_train_on_known_errors = {}".format(re_train_on_known_errors))
	print("epochs = {}".format(epochs))
	print("model = {}".format(model))
	print("############################")


def print_results(zr_acc, model_acc, b_zr_acc, b_model_acc):
	"""
	Prints the accuracy for the zr and keras models in both multi class and boolean form
	"""
	print("############################")
	print("######  Multi-Class  #######")
	print("zr acc = {}".format(zr_acc))
	print("model acc = {}".format(model_acc))
	print("########  Boolean  #########")
	print("zr acc = {}".format(b_zr_acc))
	print("model acc = {}".format(b_model_acc))


def print_results_and_cm(zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm):
	"""
	Prints the accuracy for the zr and keras models in both multi class and boolean form,
	together with the appropriate confusion matricies
	"""
	print("############################")
	print("######  Multi-Class  #######")
	print("Zero Rule Confusion Matrix")
	print(zr_cm)
	print("Model Confusion Matrix")
	print(model_cm)
	print("zr acc = {}".format(zr_acc))
	print("model acc = {}".format(model_acc))
	print("########  Boolean  #########")
	print("Zero Rule Confusion Matrix")
	print(b_zr_cm)
	print("Model Confusion Matrix")
	print(b_model_cm)
	print("zr acc = {}".format(b_zr_acc))
	print("model acc = {}".format(b_model_acc))


def generate_results(zr_acc, model_acc, y_tests, y_preds, zr_y_preds):
	"""
	Converts, arranges and generates the results into the expected format
	:param zr_acc: an array of the zero rule accuracies from training
	:param model_acc:an array of the keras model accuracies from training
	:param y_tests: the true testing values
	:param y_preds: the keras model testing values
	:param zr_y_preds: the zr model testing values
	:return:
	"""
	zr_acc = np.average(zr_acc)
	model_acc = np.average(model_acc)
	y, z, m = convert_results(y_tests, y_preds, zr_y_preds)
	zr_cm = confusion_matrix(y, z)
	model_cm = confusion_matrix(y, m)
	by = boolify(y)
	bz = boolify(z)
	bm = boolify(m)
	b_zr_cm = confusion_matrix(by, bz)
	b_model_cm = confusion_matrix(by, bm)
	b_zr_acc = accuracy_score(by, bz)
	b_model_acc = accuracy_score(by, bm)

	return zr_acc, model_acc, b_zr_acc, b_model_acc, zr_cm, model_cm, b_zr_cm, b_model_cm


def write_results(zr_acc, model_acc, b_zr_acc, b_model_acc, timer, dropout, pre_process_type, conv_layer_1,
				  conv_layer_shape, _size, scalar, represent, i, output_file, epochs, model):
	"""
	Writes the results and parameters to file
	"""
	f = open(output_file, "a+")
	f.write(
		"{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
			dropout, pre_process_type, conv_layer_1, conv_layer_shape, _size,
			scalar, represent, epochs, model, timer, zr_acc, model_acc, b_zr_acc, b_model_acc, i))

	f.close()


def model_reset(model, zr):
	"""
	resets the Keras / TF model
	:param model: The model to delete
	:return: None
	"""
	clear_session()
	reset_default_graph()
	del model
	del zr
	gc.collect()

# from sys import platform
# if platform != "linux" and platform != "linux2":
# 	raise ValueError('This code is only runnable on Linux, got version: {}'.format(platform))
