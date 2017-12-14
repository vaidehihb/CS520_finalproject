import sys
import numpy as np
import samples
import profile

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def getInitialWeights(perctype, featlength):
	weights = []
	if perctype == 'multi':
		weights = np.random.randint(10, size=(featlength, 10))
	if perctype == 'binary':
		weights = np.random.randint(10, size=(featlength, 2))
	weights = np.array(weights)
	return weights

def guessPercep(feature, weights):
	dotp = np.dot(feature, weights)
	maxval = np.argmax(dotp)
	return maxval

def trainPerc(weights, features, actuallabels, maxiter, perceptype, datasize):
	weightslocal = weights
	count = 0
	errorcount = 0
	total_input = len(features)
	for i in range(maxiter):
		for label in features:
			maxval = guessPercep(label, weightslocal)
			labelval = int(actuallabels[count])
			if labelval != maxval:
				errorcount += 1
				weighttranspose = np.transpose(weightslocal)
				weighttranspose[labelval] = np.add(weighttranspose[labelval],label)
				weighttranspose[maxval] = np.subtract(weighttranspose[maxval],label)
				weightslocal = np.transpose(weighttranspose)
			count += 1
		if errorcount/total_input > 0.1:
			count = 0
			errorcount = 0
			continue
		else:
			break
	if perceptype == 'multi':
		filename = 'percepmultiweights' + str(datasize) + '.txt'
	if perceptype == 'binary':
		filename = 'percepbinaryweights' + str(datasize) + '.txt'
	np.savetxt(filename, weightslocal)

def runPercepMultiClassTrain(maxiter, datasize):
	sampleinput = samples.readInput('digitdata/trainingimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	datasize = (datasize*len(sampleinput))/100
	actuallabels = samples.readLabels('digitdata/traininglabels', 'multi')
	features = samples.featureExtraction(sampleinput[:datasize], 4, 4)
	featlength = 0
	for label in features[:datasize]:
		featlength = len(label)
		break
	weights = getInitialWeights('multi', featlength)
	trainPerc(weights, features, actuallabels[:datasize], maxiter, 'multi', datasize)

def runPercepBinaryTrain(maxiter, datasize):
	sampleinput = samples.readInput('facedata/facedatatrain', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	datasize = (datasize*len(sampleinput))/100
	actuallabels = samples.readLabels('facedata/facedatatrainlabels', 'binary')
	features = samples.featureExtraction(sampleinput[:datasize], 6, 7)
	featlength = 0
	for label in features[:datasize]:
		featlength = len(label)
		break
	weights = getInitialWeights('binary', featlength)
	trainPerc(weights, features, actuallabels[:datasize], maxiter, 'binary', datasize)

def runPercepMultiValidator(datasize=100):
	sampleinput = samples.readInput('digitdata/testimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	actuallabels = samples.readLabels('digitdata/testlabels', 'multi')
	features = samples.featureExtraction(sampleinput, 4, 4)
	
	sampleinputtain = samples.readInput('digitdata/trainingimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	datasize = (datasize*len(sampleinputtain))/100
	weights = np.loadtxt('percepmultiweights'+str(datasize)+'.txt')
	count = 0
	errorcount = 0
	for label in features:
		guess = guessPercep(label, weights)
		#print "predicted %s actual %s" % (guess, actuallabels[count])
		if guess != int(actuallabels[count]):
			errorcount += 1
		count += 1
	prederror = float(errorcount)/len(actuallabels)	
	#prederror = errorcount/len(actuallabels)	
	print "Prediction accuracy for perceptron multi class %s." % ((1-prederror)*100)

def runPercepBinaryValidator(datasize=100):
	sampleinput = samples.readInput('facedata/facedatatest', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	actuallabels = samples.readLabels('facedata/facedatatestlabels', 'binary')
	features = samples.featureExtraction(sampleinput, 6, 7)
	
	sampleinputtrain = samples.readInput('facedata/facedatatrain', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	datasize = (datasize*len(sampleinputtrain))/100
	weights = np.loadtxt('percepbinaryweights'+ str(datasize) +'.txt')
	count = 0
	errorcount = 0
	for label in features:
		guess = guessPercep(label, weights)
		#print "predicted %s actual %s" % (guess, actuallabels[count])
		if guess != int(actuallabels[count]):
			errorcount += 1
		count += 1
	prederror = float(errorcount)/len(actuallabels)	
	#prederror = errorcount/len(actuallabels)	
	print "Prediction accuracy for perceptron Binry Class %s." % ((1-prederror)*100)

if __name__ == '__main__':
	size = 100
	#profile.runctx('print(runPercepMultiClassTrain(iter,size)); print()',globals(),{'iter': 1000, 'size':size},)
#	runPercepMultiClassTrain(1000, size)
#	runPercepBinaryTrain(1000, 100)
	runPercepMultiValidator(size)
	runPercepBinaryValidator(size)