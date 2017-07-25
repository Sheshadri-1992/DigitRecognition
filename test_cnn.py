import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import sys

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

#read data and populate into files
def loadSamples(samples,fileName,digit):
	
	reader = np.genfromtxt(fileName,delimiter=',')
	x = list(reader)
	samples = []

	for i in range(len(x)):
		
		if (x[i][-1] == digit) :

			temp = x[i]
			temp[-1] = 1
			temp = list(temp)
			temp.append(digit)
			temp = np.array(temp)
					
			samples.append(temp[0:-1])
			# print samples
			# exit()			

	return samples

def testNeural(digit1_arg,digit2_arg,digit3_arg):

	#program starts here
	wI2H = "wji_final.txt"	
	wH2O = "wkj_final.txt"

	wji=np.loadtxt(wI2H)
	wkj=np.loadtxt(wH2O)

	digit1 = digit1_arg
	digit2 = digit2_arg
	digit3 = digit3_arg

	fileName = "optdigits.tes" # this is a pre-processed data set which has around 64 features

	# we can take 3 digits, lets take 5,6 and 7
	testClassDigit1 = []
	testClassDigit1 = loadSamples(testClassDigit1,fileName,digit1)
	labelDigit1 = [digit1 for x in range(len(testClassDigit1))]

	testClassDigit2 = []
	testClassDigit2 = loadSamples(testClassDigit2,fileName,digit2)		
	labelDigit2 = [digit2 for x in range(len(testClassDigit2))]

	testClassDigit3 = []
	testClassDigit3 = loadSamples(testClassDigit3,fileName,digit3)
	labelDigit3 = [digit3 for x in range(len(testClassDigit3))]

	totalInput = testClassDigit1 + testClassDigit2 + testClassDigit3
	label = labelDigit1 + labelDigit2 + labelDigit3
	label = np.array(label)

	print "digit1 :",digit1,"=>",len(labelDigit1),"| digit2 :",digit2,"=>",len(labelDigit2),"| digit3 :",digit3,"=>",len(labelDigit3)
	classCount = {}
	classCount[digit1] = 0
	classCount[digit2] = 0
	classCount[digit3] = 0

	totalInput = np.array(totalInput)
	print "testdata ",totalInput.shape
	#testing data begins
	for i in range(len(totalInput)):

		# yj is the output
		yj=sigmoid(np.dot(wji,totalInput[i])) # activation function at I to H layer

		#zk is the output
		zk=sigmoid(np.dot(wkj,yj)) # activation function at H to O layer

		# zk has 3 output labels
		maxVal = 0
		pos = 0
		
		for k in range(len(zk)):
			if (zk[k] > maxVal):
				pos  = k
				maxVal = zk[k]

		actualLabel = label[i]
		predictedLabel = 0

		if (pos==0):
			predictedLabel = digit1
		elif (pos==1):
			predictedLabel = digit2
		elif (pos==2):
			predictedLabel = digit3

		# print "actual label ",type(actualLabel)," predicated Label ", type(predictedLabel)
		if(actualLabel == predictedLabel):
			classCount[predictedLabel] = classCount[predictedLabel] + 1

	accuracy = 0		

	for x in classCount.keys():
		accuracy = accuracy + classCount[x] 

	accuracy = accuracy / float(len(totalInput))
	print accuracy
	return accuracy