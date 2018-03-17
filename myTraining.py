#imports
import csv
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import subprocess

#keras
#from keras.models import Sequential
#from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
#from keras.layers.convolutional import Convolution2D, Cropping2D
#from keras.layers.pooling import MaxPooling2D
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint, Callback
#keras backend
#from keras.backend import tf as ktf
#sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#############################################################################
# my python network_trainer


class Record:
	def __init__(self, iD, imgCenter, imgLeft, imgRight, steering, throttle, breaks, speed):
		self.id = iD
		self.imgCenter = imgCenter
		self.imgLeft = imgLeft
		self.imgRight = imgRight
		self.steering = steering
		self.throttle = throttle
		self.breaks = breaks
		self.speed = speed
		pass

	def __str__(self):
		return "\t[%s] imgsSet: %s, %s Degree" % (self.id, type(self.imgCenter), "{:.2f}".format(math.degrees(self.steering)))

	def images(self):
		return [self.imgCenter, self.imgLeft, self.imgRight]

#create record from parsed line
def createRecordFromLine(i,argsArr):
	imgCenter = argsArr[0]
	imgLeft = argsArr[1]
	imgRight = argsArr[2]
	steering = float(argsArr[3])
	throttle = float(argsArr[4])
	breaks = float(argsArr[5])
	speed = float(argsArr[6])
	return(Record(i, imgCenter, imgLeft, imgRight, steering, throttle, breaks, speed))
#parse record files to build up tickInformation
def parseRecordFiles(recordFiles):
	recordList = []
	for r in recordFiles:
		with open(r) as csvfile:
			creader = csv.reader(csvfile)
			for i, lineArgs in enumerate(creader):
				recordList.append(createRecordFromLine(i, lineArgs))
	return recordList


#create overview of recordData
def createHistograms(recordList):
	f,a = plt.subplots(2,2)
	a = a.ravel() #makes a[0,0] to a[0] and a[0,1] to a[1] etc...
	a[0].hist([r.steering for r in recordList])
	a[1].hist([r.throttle for r in recordList])
	a[2].hist([r.breaks for r in recordList])
	a[3].hist([r.speed for r in recordList])
	plt.tight_layout()
	plt.show()



#create a augmented Record by flipping the image
def verticalFlip(newId, record):
	#create copy of entry
	r = copy.copy(record) # make a shallow copy of record
	#invert images
	r.imgLeft = cv2.flip(r.imgLeft, 1)
	r.imgRight = cv2.flip(r.imgRight, 1)
	r.imgCenter = cv2.flip(r.imgCenter, 1)
	#invert steering aswell
	r.steering_angle = r.steering * -1.0
	#(self, iD, imgCenter, imgLeft, imgRight, steering, throttle, breaks, speed):
	return r

def augmentRecordData(recordList, flipProb=0.5):
	for r in recordList:
		#do a random augmentation for the record
		doFlip = np.random.rand()
		if(doFlip < flipProb):
			recordList.append(verticalFlip(len(recordList), r))
	return recordList
	#pass

#resize an image for model pipeline
def resize(img):
    return ktf.image.resize_images(img, [66, 200])

def buildModel(keep_prob1, keep_prob2, crop_top, crop_bot):

	# input layer
	model = Sequential()
	# also do normalization on input images (lambda function)
	model.add(Lambda(lambda x: ((x / 127.5) - 1), input_shape=(160,320,3), output_shape=(160,320,3)))
    #also add cropping on input image
	model.add(Cropping2D(cropping=((crop_top,crop_bot), (0,0))))
    #also resize all the images to defined size
	model.add(Lambda(resize))

	#Convolutional Layers
	# apply a 3x3 convolution with 64 output filters on a 256x256 image:
	# model = Sequential()
	# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
	# now model.output_shape == (None, 64, 256, 256)
	# add a 3x3 convolution on top, with 32 output filters:
	#model.add(Convolution2D(32, 3, 3, border_mode='same'))
	# now model.output_shape == (None, 32, 256, 256)
    
	# L1
	#older keras version
	#subsample=(2, 2) = 2x2 stride
	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), activation="relu", input_shape=(3, 160, 320)))
	#model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
	# L2
	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation="relu"))
	#model.add(Convolution2D(36, (5,5), strides=(2,2), activation="relu"))
	# L3
	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation="relu"))
	#model.add(Convolution2D(48, (5,5), strides=(2,2), activation="relu"))
	# L4
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation="relu"))
	#model.add(Convolution2D(64, (3,3), activation="relu"))
	# L5
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation="relu"))
	#model.add(Convolution2D(64, (3,3), activation="relu"))

	# Fully Connected Layers
	# FC0
	model.add(Flatten())
	# FC1
	model.add(Dense(100))
	# Dropout DROP1
	model.add(Dropout(keep_prob1))
	model.add(Activation('relu'))
	# FC2
	model.add(Dense(50))
	# Dropout DROP2
	model.add(Dropout(keep_prob2))
	model.add(Activation('relu'))
	# FC3
	model.add(Dense(10))
	# FC4
	model.add(Dense(1))
	
	# print summary
	model.summary()
	return model

def trainModel(model, trainGen, validGen, train, valid, learningRate, epochs, samplePerEpochDivider):
	# calculate model and define optimization function 
	model.compile(loss="mse", optimizer=Adam(lr=learningRate))
	# model checkpoint definition
	checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
	#define some specifics
	fakeTrainSize = len(train)/samplePerEpochDivider
	fakeValidSize = len(valid)/samplePerEpochDivider
	#do work
	#history_object = model.fit_generator(trainGen, steps_per_epoch = fakeTrainSize,
	#	validation_data = validGen, validation_steps = fakeValidSize,
	#	epochs = epochs,
	#	verbose = 1,
	#	callbacks = [checkpoint])

	#old call
	#TypeError: fit_generator() missing 2 required positional arguments: 'samples_per_epoch' and 'nb_epoch'
	history_object = model.fit_generator(trainGen, samples_per_epoch = fakeTrainSize,
		validation_data = validGen, nb_val_samples = fakeValidSize,
		nb_epoch = epochs,
		verbose = 1,
		callbacks = [checkpoint] )


    
    
	#save model
	model.save("model.h5")
	
	return history_object

#generator function for batch data iteration
#see: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
def generator( recordSet, batchSize, augment=False, 
				#augmentation_divider=16, use_angle_dep_augmentation=False, glob_augment_probs=None, glob_bins=None
				):
	
	#grab actual pythondir
	p = subprocess.Popen(["pwd"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
	actualPath, err = p.communicate()
	actualPath = actualPath.strip() + "/"
	
	#let's go ...
	N = len(recordSet)
	# loop forever, never exit the gen
	#while 1:
	#go through batches
	for offset in range(0, N, batchSize):
		batch = recordSet[offset:offset+batchSize]
		imageData = []
		angleData = []

		for i, record in enumerate(batch):
			#get path to images
			images = []
			#paths = [record.imgCenter, record.imgLeft, record.imgRight]
			paths = [record.imgLeft]
			for p in paths:
				##filename = p.split('/')[-1] #grab filename out of path
				##directory = p.split('/')[-3]
				##newPath = directory + "/IMG/" + filename
				path = actualPath + "/" + p[p.find("records"):]
				print(p)
				img = cv2.imread(path)
				#brn
				#convert to YUV
				images.append(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))

				pass
			#set loaded images to record
			#batch[i].imgCenter = images[0]
			#batch[i].imgLeft = images[1]
			#batch[i].imgRight = images[2]

			#append to list of data
			angleData.append(record.steering)
			for img in images:
				imageData.append(img)

		#do augmentation on the images
		if augment:
			pass

		#X_train = np.array(images)
		#y_train = np.array([r.steering for r in batch])
		X_train = np.array(imageData)
		y_train = np.array(angleData)
		yield shuffle(X_train, y_train)

#############################################################################
############################## Main #########################################
#############################################################################
def main():

	# static guard vars
	showHistograms = False
	doAugmentation = True

	#training parameters
	epochs = 5
	batchSize = 128
	learningRate = 0.0001
	samplePerEpochDivider = 5 # for reducing computational work

	#model parameters:
	keepProb1 = 0.50
	keepProb2 = 0.25
	cropTop = 70
	cropBot = 25

	# read the driving logs of all measurements
	#recordFiles = [	"./records/record_1_f/driving_log.csv",
	#				"./records/record_1_b/driving_log.csv",
	#				"./records/record_2_f/driving_log.csv",
	#				"./records/record_2_b/driving_log.csv"
	#				]
	recordFiles = [	"./records/record_thm/driving_log.csv"
					]
	# parse records
	print("[*] Parsing Records ...")
	print("\t%s" % recordFiles)
	recordList = parseRecordFiles(recordFiles)
	#for r in recordData:
	#	print(r)

	# create hisogram
	# https://stackoverflow.com/questions/20174468/how-to-create-subplots-of-pictures-made-with-the-hist-function-in-matplotlib-p
	if(showHistograms):
		print("[*] Creating Histograms ...")
		createHistograms(recordList)

	# do some augmentation of the initial data if needed
	# we dont want this here, because we would have to get all images into ram ...
	# so we want to do this later while batching and learning the data
	# if(doAugmentation):
	# 	recordList = augmentRecordData(recordList)
	# one augmentation we can do is shuffle the input data, so that our train, validation and test set is allways different
	print("[*] Shuffling records ...")
	recordList =  shuffle(recordList)	# sklearn tool

	# split our data into seperate thingys
	train, valid = train_test_split(recordList, test_size = 0.25) #sklearn tool
	#
	print("[*] Splitted record set ...")
	print("\ttrain: %s, validation: %s" % (len(train), len(valid) ))

	print("[*] Creating Generators for record set ...")
	trainGen = generator(train, batchSize=batchSize, augment=False)
	validGen = generator(valid, batchSize=batchSize)
	print(trainGen[0])
	print("...")
	# build model: from NVIDIA paper with dropout
	#print("[*] Building Training Model ...")
	# model = buildModel(keepProb1, keepProb2, cropBot, cropTop)

	# # train the model
	# hObj = trainModel(	model, trainGen, 
	# 					validGen,
	# 					train,
	# 					valid,
	# 					learningRate, 
	# 					epochs,
	# 					samplePerEpochDivider)

	#plot the training and validation results
	#plt.plot(hObj.history['loss'])
	#plt.plot(hObj.history['val_loss'])
	#plt.title('model loss')
	#plt.ylabel('MSE Loss')
	#plt.xlabel('Epoch')
	#plt.legend(['training set', 'validation set'], loc='bottom right')
	#plt.show()
	
if __name__ == '__main__':
	main()