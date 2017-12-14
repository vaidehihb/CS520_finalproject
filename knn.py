import sys
import numpy as np
from numpy import reshape
import samples
import scipy
from scipy.spatial.distance import cdist
import profile


DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def knnfeatureExtract(sampleinput):
	knnfeatures = []
	for label in sampleinput:
		lab = list(label.flatten())
		knnfeatures.append(lab)
	knnfeatures = np.array(knnfeatures)	
	return knnfeatures


def findKNeighbours(sample, trainfeatures, actuallabels, k):
	dist = scipy.spatial.distance.cdist(sample, trainfeatures, 'euclidean')
	sorteddist = np.argsort(dist[0])	

	kneighbours = []
	for i in range(k):
		if i < len(sorteddist):
			kneighbours.append(int(actuallabels[sorteddist[i]]))
	
	kneighbours = np.array(kneighbours)
	counts = np.bincount(kneighbours)
	return np.argmax(counts)
	

def testknn(testfeatures, testactuallabels, trainfeatures, trainactuallabels, k):
	errorcount = 0
	count = 0
	
	for sample in testfeatures:
		profile.runctx('print(findKNeighbours(sample, trainfeatures, trainactuallabels, k)); print()',globals(),{'sample': [sample], 'trainfeatures': trainfeatures, 'trainactuallabels': trainactuallabels, 'k':k },)
		prediction = findKNeighbours([sample], trainfeatures, trainactuallabels, k)
		if int(prediction) != int(testactuallabels[count]):
			errorcount += 1
		count += 1
	prederror = float(errorcount)/len(testfeatures)
	print "Prediction accuracy for KNN %s" % ((1-prederror)*100)

def runKnnTesterMulti(datasize=100):
	print 'Running KNN MultiClass for datasize %s' % (datasize)
	testsampleinput = samples.readInput('digitdata/testimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	testactuallabels = samples.readLabels('digitdata/testlabels', 'multi')	
	testfeatures = knnfeatureExtract(testsampleinput[:datasize])
	#testfeatures = samples.featureExtraction(testsampleinput, 4, 4)
	trainsampleinput = samples.readInput('digitdata/trainingimages', DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
	datasize = (datasize*len(trainsampleinput))/100
	trainactuallabels = samples.readLabels('digitdata/traininglabels', 'multi')
	#trainfeatures = samples.featureExtraction(trainsampleinput, 4, 4)
	trainfeatures = knnfeatureExtract(trainsampleinput[:datasize])
	testknn(testfeatures, testactuallabels, trainfeatures, trainactuallabels[:datasize], 5)

def runKnnTesterBinary(datasize=100):
	print 'Running KNN BinaryClass for datasize %s' % (datasize)
	testsampleinput = samples.readInput('facedata/facedatatest', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	testactuallabels = samples.readLabels('facedata/facedatatestlabels', 'binary','knn')	
	testfeatures = knnfeatureExtract(testsampleinput)
	#testfeatures = samples.featureExtraction(testsampleinput, 6, 7)
	trainsampleinput = samples.readInput('facedata/facedatatrain', FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
	datasize = (datasize*len(trainsampleinput))/100
	trainactuallabels = samples.readLabels('facedata/facedatatrainlabels', 'binary', 'knn')
	trainfeatures = knnfeatureExtract(trainsampleinput[:datasize])
	#trainfeatures = samples.featureExtraction(trainsampleinput, 6, 7)
	print('Running KNN binaryclass \n')
	testknn(testfeatures, testactuallabels, trainfeatures, trainactuallabels[:datasize], 5)

if __name__ == '__main__':
	runKnnTesterMulti(100)
	runKnnTesterBinary(100)
