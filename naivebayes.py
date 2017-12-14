import sys
import numpy as np
import samples
import math
import profile

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def getLabelPosition(trainlabels, classtype):
	trainlabels = np.array(trainlabels)
	labelpos = []
	if classtype == 'multi':
		total = list(range(10))
	if classtype == 'binary':
		total = list(range(2))
	for i in total:
		position = list(np.where(trainlabels == i)[0])
		labelpos.append(position)
	return np.array(labelpos)

def getLabelDetails(trainlabels, classtype, datasize):
	trainlabels = np.array(trainlabels)
	trainsize = len(trainlabels)
	labeldetails = []
	if classtype == 'multi':
		total = list(range(10))
		probfilename = 'naiveprobmulti' + str(datasize) + '.txt'
	if classtype == 'binary':
		total = list(range(2))
		probfilename = 'naiveprobbinary' + str(datasize) + '.txt'
	for i in total:
		position = list(np.where(trainlabels == i)[0])
		row = [float(i), float(len(position)), float(len(position))/trainsize]
		labeldetails.append(row)
	labeldetails = np.array(labeldetails)
	np.savetxt(probfilename, labeldetails)

def naiveBayesTrain(labelpos, trainfeatures, testfeature, trainprobs):
	filoopcount = 0
	trainfeaturestrans = np.transpose(trainfeatures)
	ficount = 0

	conditionalprobabilities = []
	labelprob = []

	for label in labelpos:
		for fi in testfeature:		 
			for pos in label:
				if trainfeaturestrans[filoopcount][pos] == fi:
					ficount += 1
			if ficount == 0:
				prob = (float(ficount)+float(1))/len(label)
			else:
				prob = float(ficount)/len(label)
			labelprob.append(prob)
			filoopcount += 1
			ficount = 0
		conditionalprobabilities.append(labelprob)
		filoopcount = 0
		labelprob = []

	ycount = 0
	guesslist = []
	sum = 0
	for y in conditionalprobabilities:
		for fi in y:
			sum += math.log(fi)
		sum += math.log(trainprobs[ycount])
		guesslist.append(sum)
		sum = 0
		ycount += 1

	return np.argmax(guesslist)

def runNaiveBayesMultiTest(datasize= 100):
	trainsampleinput = samples.readInput('digitdata/trainingimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	trainlabels = samples.readLabels('digitdata/traininglabels', 'multi')
	datasize = (datasize*len(trainsampleinput))/100
	labelpos = getLabelPosition(trainlabels[:datasize], 'multi')
	trainfeatures = samples.featureExtraction(trainsampleinput[:datasize], 4, 4)
	
#	getLabelDetails(trainlabels, 'multi', datasize)
	trainprobs = np.loadtxt("naiveprobmulti" + str(datasize) + ".txt")
	trainprobstrans = np.transpose(trainprobs)

	testsampleinput = samples.readInput('digitdata/testimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	testlabels = samples.readLabels('digitdata/testlabels', 'multi')
	testfeatures = samples.featureExtraction(testsampleinput, 4, 4)

	count = 0
	errorcount = 0
	for sample in testfeatures:
#		profile.runctx('print(naiveBayesTrain(labelpos, trainfeatures, sample, trainprobstrans)); print()',globals(),{'labelpos': labelpos, 'trainfeatures': trainfeatures, 'sample': sample, 'trainprobstrans':trainprobstrans[2] },)
		prediction = naiveBayesTrain(labelpos, trainfeatures, sample, trainprobstrans[2])
#		print "predicted %s actual %s" % (prediction, testlabels[count])
		if prediction != testlabels[count]:
			errorcount += 1
		count += 1
	prederror = float(errorcount)/len(testlabels)
	print "Prediction accuracy for Naive Bayes MultiClass %s." % ((1-prederror)*100)

def runNaiveBayesBinaryTest(datasize):
	trainsampleinput = samples.readInput('facedata/facedatatrain', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	datasize = (datasize*len(trainsampleinput))/100
	trainlabels = samples.readLabels('facedata/facedatatrainlabels', 'binary')
	labelpos = getLabelPosition(trainlabels[:datasize], 'binary')
	trainfeatures = samples.featureExtraction(trainsampleinput[:datasize], 6, 7)

	getLabelDetails(trainlabels, 'binary', datasize)
	trainprobs = np.loadtxt("naiveprobbinary" + str(datasize) + ".txt")
	trainprobstrans = np.transpose(trainprobs)

	testsampleinput = samples.readInput('facedata/facedatatest', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	testlabels = samples.readLabels('facedata/facedatatestlabels', 'binary')
	testfeatures = samples.featureExtraction(testsampleinput, 6, 7)

	count = 0
	errorcount = 0
	for sample in testfeatures:
#		profile.runctx('print(naiveBayesTrain(labelpos, trainfeatures, sample, trainprobstrans)); print()',globals(),{'labelpos': labelpos, 'trainfeatures': trainfeatures, 'sample': sample, 'trainprobstrans':trainprobstrans[2] },)
		prediction = naiveBayesTrain(labelpos, trainfeatures, sample, trainprobstrans[2])
#		print "predicted %s actual %s" % (prediction, testlabels[count])
		if prediction != testlabels[count]:
			errorcount += 1
		count += 1
	prederror = float(errorcount)/len(testlabels)
	print "Prediction accuracy for Naive Bayes Binary Class data %s." % ((1-prederror)*100)

if __name__ == '__main__':
	runNaiveBayesMultiTest(100)
	runNaiveBayesBinaryTest(100)