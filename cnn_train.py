import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import sys
import math
import test_cnn
import plotGraph
#train any 3 digits, the data link is here
# 

# sigmoid function
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))


#Returns index based on the label / digit
def mapIndex(labelArg):
	
	if labelArg==digit1:
		return 0
	elif labelArg==digit2:
		return 1
	else:
		return 2


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

fileName = "optdigits.tra" # this is a pre-processed data set which has around 64 features
# we can take 3 digits, lets take 5,6 and 7
digit1 = 0
digit2 = 5
digit3 = 9

trainDigit1 = []
trainDigit1 = loadSamples(trainDigit1,fileName,digit1)
labelDigit1 = [digit1 for x in range(len(trainDigit1))]

trainDigit2 = []
trainDigit2 = loadSamples(trainDigit2,fileName,digit2)		
labelDigit2 = [digit2 for x in range(len(trainDigit2))]

trainDigit3 = []
trainDigit3 = loadSamples(trainDigit3,fileName,digit3)
labelDigit3 = [digit3 for x in range(len(trainDigit3))]


totalInput = trainDigit1 + trainDigit2 + trainDigit3
label = labelDigit1 + labelDigit2 + labelDigit3
label = np.array(label)
print "chosen digit1 ",digit1," ",len(labelDigit1)
print "chosen digit2 ",digit2," ",len(labelDigit2)
print "chosen digit3 ",digit3," ",len(labelDigit3)
print len(totalInput)


def trainNeural(nH_arg,nI_arg):
	#this concludes gathering of data, the data has 64 features (remember 8x8 presented in the assignment question)
	# so the input layer has 64 inputs, this input layer has to be trained a total of (no of 5 + no of 6 + no of 7) number of times
	# the number of hidden units have to be decided
	# the number of output units have to be decided

	nH = nH_arg
	nI = nI_arg #64 + 1 for bias
	# b2H = np.random(nI)*0.03 #bias to Hidden Layer
	# b2O = np.random(nH)*0.03 #bias to Output Layer

	wji=np.zeros((nH,nI)) # here nH is the number of hidden units, trail and error, 64 is the input units (8x8)
	wkj=np.zeros((3,nH)) # here 3 is the output , we expect 3 outputs since in the question they have given any 3 digits, nH is the number of hidden units

	for i in range(nH): # initialize all the weights from Input to Hidden randomly
		wji[i]=np.random.rand(nI)*0.03
	for i in range(3):  # initialize all the weights from Hidden to Output randomly
		wkj[i]=np.random.rand(nH)*0.03	

	err=1000000000 # initialize error to some high value
	theta=0.1 #allowed error value
	eta=0.01 # learning rate
	iterations =1 # stop after some iterations, to avoid infinite loops
	e=5

	while math.fabs(e)>theta:
		prev=err
		err=0
		
		for i in range(len(totalInput)):
			a=np.zeros((nI,1)) # since we have 64 features in the input
			for k in range(nI):
				a[k][0]=totalInput[i][k]

			#wji stands for all the input unit vectors
			# print totalInput[i].shape
			yj=sigmoid(np.dot(wji,totalInput[i])) # activation function at I to H layer
			yj[-1] = 1

			#zk is the output
			zk=sigmoid(np.dot(wkj,yj)) # activation function at H to Output layer

			#print zk.shape
			tk=np.zeros((3,)) # 3 labels for output 

			index = mapIndex(label[i])
			
			tk[index]=1
			
			temp=tk-zk

			err=err+(temp.dot(temp))*0.5  # this is 1/2(t - z)^2

			deltak=np.dot(temp,np.dot((1-zk),zk)) # z = f(netk) 
			
			deltaj=np.dot(np.dot((1-yj),yj),np.dot(deltak,wkj))
			
			tempdeltaj=np.zeros((nH,1))
			for k in range(nH):
				tempdeltaj[k][0]=deltaj[k]
			
			wji=wji+(eta*np.dot(tempdeltaj,a.T)) #page 292 of the book
			
			tempy=np.zeros((1,nH))
			for k in range(nH): #Each node in the hidden layer will have a value yj
				tempy[0][k]=yj[k]

			tempdeltak=np.zeros((3,1))
			for k in range(3):
				tempdeltak[k][0]=deltak[k]

		
			wkj=wkj+(eta*np.dot(tempdeltak,tempy))
		
		# print "Itr : ", iterations," error " , err, " e is ",math.fabs(e)

		iterations=iterations+1

		if iterations>10000:
			break

		e=prev-err
	print "e after coming out ",e	

	np.savetxt('wji.txt',wji)
	np.savetxt('wkj.txt',wkj)

	accuracy = test_cnn.testNeural(digit1,digit2,digit3)
	return accuracy

# this is where the code starts
def mainFunction():

	inputUnits = 65 #(fixed)
	nHlist = [31,41,51,61,71,81,91,101,111,121,131,141,151]
	accuracyList = []

	with open('results.csv','wb') as csvfile :

		writer = csv.writer(csvfile, delimiter=',')
		temp = []

		for ele in range(1):
			accuracy = trainNeural(151,inputUnits)
			temp.append(accuracy)
			writer.writerow(temp)
			accuracyList.append(accuracy)

		newList = [x-1 for x in nHlist]
		plotGraph.plotMyGraph([151],temp,digit1,digit2,digit3)

	print "final accuracy ",accuracyList



#call to main function
mainFunction()			