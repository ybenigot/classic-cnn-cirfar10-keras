# adapted from keras examples
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping


import numpy as np
import sys
import imageutils as im
import time
import cifar10labels as cl
import random as rn
import datetime as dt
import os
import plot as pl

image_size=32
image_border=8
input_size=image_size#+2*image_border
batch_size_param=256
maps_count_param=128
learn_rate=0.003
decay_param=4e-5 # 1 - 0.1**(1/(100*45000/256))) 
lambda_reg=0.0005
training_set_ratio=0.9

#import keras.layers.advanced_activations as ka
#ka.LeakyReLU(alpha=0.4

def make_model():
	''' define the model'''
	model = Sequential()
	# input: 32x32 images with 3 channels -> (3, 32, 32) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(maps_count_param, 3, 3, border_mode='same', input_shape=(3, input_size, input_size),init='he_normal',W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
	model.add(Convolution2D(maps_count_param, 3, 3, border_mode='same', init='he_normal',W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
#	model.add(Dropout(0.3))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(maps_count_param*2, 3, 3, border_mode='same', init='he_normal',W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
#	model.add(Dropout(0.3))
	model.add(Convolution2D(maps_count_param*2, 3, 3, border_mode='same', init='he_normal',W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
#	model.add(Dropout(0.3))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(maps_count_param*4, 3, 3, border_mode='same', init='he_normal',W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
#	model.add(Dropout(0.3))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(2048,W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(1024,W_regularizer=l2(lambda_reg)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(10,W_regularizer=l2(lambda_reg)))
	model.add(Activation('softmax'))
#	model.add(Dropout(0.5))

	sgd = SGD(lr=learn_rate, decay=decay_param, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])

	print('model parameters:',model.count_params())
	print('model characteristics:',model.summary())
	print('----------------------------------------------------------------------------------------')

	return model

def load_data():
	''' load and normalize data from dataset files '''
	(X, y), (X_test, y_test) = cifar10.load_data()
	n = X.shape[0]
	n1 = int(n * training_set_ratio)
	X_train=X[0:n1,:]
	y_train=y[0:n1]
	X_val=X[n1:n,:]
	y_val=y[n1:n]
	return X_train, y_train, X_val, y_val, X_test, y_test

def scale_data(data):
	# scale the image pixel values into interval 0,2, mean will be substrated later
	scale=128
	n=data.shape[0]
	data = data.astype('float32')
	data = data.reshape((n,3,image_size,image_size))
	data /= scale
	# extend image size with zeroes
	#data2 = np.zeros((n,3,input_size,input_size),dtype=np.float32)
	#for i in range(0,n):
	#	for j in range(0,3):
	#		data2[i,j,image_border:image_size+image_border,image_border:image_size+image_border] = data[i,j,:,:]
	#return data2	
	return data

def preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test):
	print('start preprocess...')

	X_train=scale_data(X_train)
	X_val=scale_data(X_val)
	X_test=scale_data(X_test)

	#substract mean, per sample and per color channel 
	X_train, X_val, X_test = im.mean2(X_train, X_val, X_test)

	#apply ZCA whitening on each color channel
	#X_train=im.whiten(X_train,epsilon=0.1)
	#X_test=im.whiten(X_test,epsilon=0.1)

	g = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,\
	fill_mode='nearest',dim_ordering='th') 
	g.fit(X_train)
	
	y_train = to_categorical(y_train)
	y_val = to_categorical(y_val)
	y_test = to_categorical(y_test)

	print('...done')

	return g, X_train, y_train, X_val, y_val, X_test, y_test

def fit(model , g, X_train, y_train, X_val, y_val, epochs):
	''' train the model '''
	#history = model.fit(X_train, Y_train, batch_size=batch_size_param, nb_epoch=epochs,validation_split=0.1)
	earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
	history=model.fit_generator(g.flow(X_train, y_train, batch_size=batch_size_param),callbacks=[earlyStopping], \
		samples_per_epoch=len(X_train),nb_epoch=epochs,verbose=1, validation_data=(X_val, y_val))

	return history

def predict(model,X,y):
	''' predict Y given X using model '''
	pred = model.predict(X, batch_size=batch_size_param, verbose=0)
	#g.fit(X)
	#pred = model.predict_generator(g.flow(X, y, batch_size=512), X.shape[0])
	return pred

def compute_accuracy(pred,Y):
	'''compute prediction accuracy by matching pred and Y'''
	comparison = np.argmax(pred,1)==np.argmax(Y,1)
	accuracy = sum(comparison)/pred.shape[0]
	return accuracy

def show_results(pred,X,Y):
	classification=np.argmax(pred,1)	
	for i in rn.sample(range(X.shape[0]), 1):
		im.display_normalized_image(X[i,:],input_size)
		print('prediction:',cl.labels[classification[i]],'actual value:',cl.labels[np.argmax(Y[i])])
		time.sleep(5)

def main():

	epochs=int(sys.argv[1])
	print(epochs,' epochs')

	X_train, y_train, X_val, y_val, X_test, y_test = load_data()

	for i in range(0,1):
		im.display_image(X_train[i,:],image_size)

	g, X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test)
	print('X_train.shape ',X_train.shape,'y_train.shape ',y_train.shape)
	print('X_val.shape ',  X_val.shape,  'y_val.shape ',  y_val.shape)
	print('X_test.shape ', X_test.shape, 'y_test.shape ', y_test.shape)

	# learn the model
	model=make_model()
	hist=fit(model , g, X_train, y_train, X_val, y_val, epochs)
	print(hist.history)

	# test the model
	pred = predict(model,X_test,y_test)
	accuracy=compute_accuracy(pred,y_test)
	print('accuracy on test data: ',accuracy*100, '%')
	show_results(pred,X_test,y_test)

	# save learned weights
	f="%d-%m-%y"
	filename='record/weights-'+dt.date.today().strftime(f)
	model.save_weights(filename,overwrite=True)

	pl.plot(hist.history,len(hist.history['acc']))
	os.system('./plot.sh')


if __name__ == "__main__":
    main()


