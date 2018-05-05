import re
from porter2stemmer import Porter2Stemmer
from random import shuffle
import math
import numpy as np
import matplotlib.pyplot as plt

##### Stop words
stoplist = []
f = open("/home/sree/Machine-Learning-course/Assignment2/stopwords.txt",'r')
for line in f: 
	stoplist.append(line.replace("\n",""))
#print stoplist

stemmer = Porter2Stemmer()


##### Remove stop words from messages and form 2-d list
wosw = []
f = open("/home/sree/Machine-Learning-course/Assignment2/Assignment_2_data.txt",'r')
for line in f: 
	z = re.split('\t| |;|,|\*|\n|\.',line)
	b = filter(lambda a: a != '', z)
	l3 = [stemmer.stem(x) for x in b if x.lower() not in stoplist]
	wosw.append(l3)
shuffle(wosw)
#print wosw

##### Make a list of tokens
tokens = []
for row in wosw:
    for item in row:
        if item != 'ham' and item != 'spam':
        	tokens.append(item)
tokens = sorted(set(tokens))
tokens = tokens[:2000]
#print tokens

#####  One hot encoding
##### ham --> 0 , spam --> 1
encoding = []
for row in wosw:
	sub_enc = []
	for word in tokens:
		if row.count(word) == 0:
			sub_enc.append(0)
		else:
			sub_enc.append(1)
	if row[0] == 'ham' :
		sub_enc.append(0)
	else:
		sub_enc.append(1)
	encoding.append(sub_enc)
#print encoding

##### Split into train and test set
len_part=int(math.ceil(len(encoding)*0.8))
training = encoding[:len_part]
test = encoding[len_part:]

##### Forward propagation at a single neuron
def forward_propagte(weights, inputs):
	return sigmoid_activate(np.dot(inputs,weights))
 
##### Neuron activation function
def sigmoid_activate(convolution):
	return 1/(1+np.exp(-convolution))

def tanh_activate(convolution):
	return np.tanh(convolution)

##### Derivative function
def sigmoid_derivative(output):
	return output * (1.0 - output)

def tanh_derivative(output):
	return  (1.0 - (output ** 2))

def in_sample(W1,W2,W3):
	test_error = 0.0
	misclass = 0

	testx = np.asarray(training)
	instance1 = np.asarray(training)
	test1 = np.ones(instance1.shape[0],dtype=int)
	instance1[:, -1] = test1
	x1 = forward_propagte(W1,instance1)
	test1 = np.ones((x1.shape[0],1),dtype=int)
	x1 = np.append(x1,test1,axis = 1)
	x2 = forward_propagte(W2,x1)
	test1 = np.ones((x2.shape[0],1),dtype=int)
	x2 = np.append(x2, test1, axis=1)
	x3 = np.dot(x2,W3)
	z_exp = np.exp(x3)
	sum_z_exp = np.sum(z_exp,axis = 1,keepdims = True)
	softmax = z_exp/sum_z_exp
	testxx = testx[:,-1]
	x = np.ones((softmax.shape[0],1),dtype=int)

	for count in range(0,softmax.shape[0]): 
		if softmax[count][0] > 0.5:
			x[count][0] = 0
		else:
			x[count][0] = 1
		if x[count][0] != testxx[count] :
	 		misclass = misclass + 1
	 	#print x[count][0],testxx[count] 
		test_error = test_error + 0.5 * ((x[count][0] - testxx[count]) ** 2)
	return [misclass,test_error] 

def out_of_sample(W1,W2,W3,final):
	test_error = 0.0
	misclass = 0

	testx = np.asarray(test)
	instance1 = np.asarray(test)
	test1 = np.ones(instance1.shape[0],dtype=int)
	instance1[:, -1] = test1
	x1 = forward_propagte(W1,instance1)
	test1 = np.ones((x1.shape[0],1),dtype=int)
	x1 = np.append(x1,test1,axis = 1)
	x2 = forward_propagte(W2,x1)
	test1 = np.ones((x2.shape[0],1),dtype=int)
	x2 = np.append(x2, test1, axis=1)
	x3 = np.dot(x2,W3)
	z_exp = np.exp(x3)
	sum_z_exp = np.sum(z_exp,axis = 1,keepdims = True)
	softmax = z_exp/sum_z_exp
	testxx = testx[:,-1]
	x = np.ones((softmax.shape[0],1),dtype=int)

	for count in range(0,softmax.shape[0]): 
		if softmax[count][0] > 0.5:
			x[count][0] = 0
		else:
			x[count][0] = 1
		if x[count][0] != testxx[count] :
	 		misclass = misclass + 1
	 	if final == 1:
	 		print x[count][0],testxx[count] 
		test_error = test_error + 0.5 * ((x[count][0] - testxx[count]) ** 2)
	return [misclass,test_error] 

W1 = np.random.normal(0,1,(len(tokens)+1, 100)).astype(np.float32) * np.sqrt(2.0/(len(tokens)+1))
W2 = np.random.normal(0,1,(101,50)).astype(np.float32) * np.sqrt(2.0/(101))
W3 = np.random.normal(0,1,(51,2)).astype(np.float32) * np.sqrt(2.0/(51))
LR = 0.1
etr = []
ete = []
y = []
i = 0
e1 = 0
cummmisclassout = 0
cummmisclassin = 0
cummerrorout = 0
cummerrorin = 0

print "Training started"
epochs = 1
while True:
	print "epoch ",epochs
	shuffle(training)
	for instance in training:
		i = i + 1
		instance1 = instance[:len(instance)-1]
		instance1.append(1)
		instance1 = np.array([instance1])
		#print instance1.shape
		#print W1.shape
		x1 = forward_propagte(W1,instance1)
		x1 = np.append(x1, 1)
		x1 = np.array([x1])

		#print W2.shape
		x2 = forward_propagte(W2,x1)
		x2 = np.append(x2, 1)
		x2 = np.array([x2])

		x3 = np.dot(x2,W3)
		z_exp = np.exp(x3)
		sum_z_exp = z_exp[0][0] + z_exp[0][1]
		softmax = z_exp/sum_z_exp

		if instance[-1] == 0:
		    actual = [1,0]
		else:
		    actual = [0,1]
		l1 = 0.5 * ((softmax[0][0] - actual[0]) ** 2)
		l2 = 0.5 * ((softmax[0][1] - actual[1]) ** 2)

		del3dash = []
		dell1 = ((softmax[0][0] - actual[0]) - (softmax[0][1] - actual[1])) * softmax[0][0] * softmax[0][1]
		del3dash.append(dell1)
		del3dash.append(-dell1)
		del3 = np.asarray([del3dash])

		temp3 = np.dot(del3,W3.T)
		del2 = np.dot(np.dot(x2,(1-x2).T),temp3)

		temp2 = np.dot(W2,del2[:,0:del2.shape[1]-1].T)
		del1 =  np.dot(np.dot(x1,(1-x1).T),temp2.T)

		W3 = W3 - LR * del3 * x2.T
		W2 = W2 - LR * np.dot(x1.T,del2[:,0:del2.shape[1]-1])
		W1 = W1 - LR * np.dot(instance1.T,del1[:,0:del1.shape[1]-1])
	print "here"
	[misclass,test_error] = out_of_sample(W1,W2,W3,0)
	print "out sample error ", test_error , " out sample misclass ", misclass,"/",len(test)
	ete.append(test_error)
	[misclass,test_error] = in_sample(W1,W2,W3)
	print "in sample error ", test_error , " in sample misclass ", misclass,"/",len(training)
	etr.append(test_error)
	y.append(epochs)

	if test_error < 20:
		break
	epochs = epochs +1

print "training error"
print len(etr)
print "test error"
print len(ete)
print "epochs"
print len(y)

out_of_sample(W1,W2,W3,1)

plt.plot(y,etr,'-',label='Insample Error')
plt.plot(y,ete,'-',label='Outsample Error')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error vs Epochs in part2")
plt.legend()
plt.show()