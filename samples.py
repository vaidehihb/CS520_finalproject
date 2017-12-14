import sys
import numpy as np

def readInput(filepath, width, height):
	f = open(filepath, 'r')
	inputstream = []
	for line in f:
		inputstream.append(line.rstrip('\n'))
	sampleinput = []
	label = []
	row = []
	count = 0
	image = 1
	for line in inputstream:	
		for j in line:
			if j == ' ':
				row.append(0)
			if j == '#':
				row.append(1)
			if j == '+':
				row.append(2)
		count += 1
		if count < height:
			label.append(row)
		if count == height: 
			label.append(row)
			sampleinput.append(label)
			label = []
			count = 0	
			image += 1					
		row = []

	sampleinput = np.array(sampleinput)
	#size = int((percent/100)*len(sampleinput))
	return sampleinput

def readLabels(filepath, labeltype, classtype = 'noknn'):
	f = open(filepath, 'r')
	inputstream = []
	for line in f:
		if classtype == 'binary':
			if labeltype != 'knn':
				if int(line.rstrip('\n')) == 0:
		 			inputstream.append(-1)
		 		else:
		 			inputstream.append(int(line.rstrip('\n')))
		 	else:
		 		inputstream.append(int(line.rstrip('\n')))
		else:
			inputstream.append(int(line.rstrip('\n')))
	return inputstream

def featureExtraction(sampleinput, gridwidth, gridheight):
	features = []
	for label in sampleinput:
		rows = getBlackPixelsCount(label, 'row')
		columns = getBlackPixelsCount(label, 'col')
		total = getBlackPixelsCount(label, 'total')
		griddata = getGridPixelCount(label, gridwidth, gridheight)
		feat = []
		feat.append(total)
		feat = feat + rows + columns + griddata
		features.append(feat)
	features = np.array(features)
	return features	

def getBlackPixelsCount(label, feattype):
	count = 0
	data = []
	if feattype == 'row':
		for row in label:
			for i in row:
				if i > 0:
					count += 1
			data.append(count)
			count = 0		
	if feattype == 'col':
		label = np.transpose(label)
		data = getBlackPixelsCount(label, 'row')
	if feattype == 'total':
		rows = getBlackPixelsCount(label, 'row')
		data = 0
		for i in rows:
			data += i
	return data	

def getGridPixelCount(label, gridwidth, gridheight):
	griddata = []
	x,y = label.shape
	blockwidth = int(x/gridwidth)
	blockheight = int(y/gridheight)
	for i in range(gridheight):
		for j in range(int(i*blockheight),x,blockwidth):
			block = label[j:int(j+blockwidth-1), j:int(j+blockheight-1)]
			blocktotal = getBlackPixelsCount(block, 'total')
			griddata.append(blocktotal)
	return griddata