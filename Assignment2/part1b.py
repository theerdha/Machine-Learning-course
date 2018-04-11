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

def out_of_sample(W1,W2,W3):
	test_error = 0.0
	misclass = 0
	for instance2 in test:
		instance1 = instance2[:len(instance2)-1]
		instance1.append(1)
		instance1 = np.array([instance1])
		x1 = forward_propagte(W1,instance1)
		x1 = np.append(x1, 1)
		x1 = np.array([x1])
		x2 = forward_propagte(W2,x1)
		x2 = np.append(x2, 1)
		x2 = np.array([x2])
		h = forward_propagte(W3,x2)
		if(h[0] > 0.5):
			x = 1
		else:
			x = 0
		#print h[0],x,instance2[-1]
		if x != instance2[-1] :
			misclass = misclass + 1
		test_error += 0.5 * ((h[0] - instance2[-1]) ** 2)

	print "out of sample error is ",test_error
	print "out of sample misclass ", misclass 
	return test_error 

def in_sample(W1,W2,W3):
	test_error = 0.0
	misclass = 0
	for instance2 in training:
		instance1 = instance2[:len(instance2)-1]
		instance1.append(1)
		instance1 = np.array([instance1])
		x1 = forward_propagte(W1,instance1)
		x1 = np.append(x1, 1)
		x1 = np.array([x1])
		x2 = forward_propagte(W2,x1)
		x2 = np.append(x2, 1)
		x2 = np.array([x2])
		h = forward_propagte(W3,x2)
		if(h[0] > 0.5):
			x = 1
		else:
			x = 0
		#print h[0],x,instance2[-1]
		if x != instance2[-1] :
			misclass = misclass + 1
		test_error += 0.5 * ((h[0] - instance2[-1]) ** 2)

	print "insample error is ",test_error
	print "insample misclass ", misclass 
	return test_error 


W1 = np.random.randn(len(tokens)+1, 100).astype(np.float32) * np.sqrt(2.0/(len(tokens)+1))
W2 = np.random.randn(101,50).astype(np.float32) * np.sqrt(2.0/(101))
W3 = np.random.randn(51,1).astype(np.float32) * np.sqrt(2.0/(51))
LR = 1e-6
etr = []
ete = []
y = []
i = 0

print "Training started"
epochs = 100
while epochs > 0:
	for instance in training:
		# print type(W1)
		i = i + 1
		y.append(i)
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
		h = forward_propagte(W3,x2)

		e = 0.5 * ((h[0] - instance[-1]) ** 2)
		print h[0],instance[-1]
		print "error ",e
		# if abs(e1 - e) < 0.000001 :
		# 	break
		del3 = np.asscalar((h[0] - instance[-1]) * sigmoid_derivative(h[0]))

		temp3 = del3*W3.T
		del2 = np.dot(x2,(1-x2).T) * temp3
		
		temp2 = np.dot(W2,del2[:,0:del2.shape[1]-1].T)
		del1 =  np.dot(x1,(1-x1).T) * temp2.T


		W3 = W3 - LR * del3 * x2.T
		W2 = W2 - LR * np.dot(x1.T,del2[:,0:del2.shape[1]-1])
		W1 = W1 - LR * np.dot(instance1.T,del1[:,0:del1.shape[1]-1])
		test_error = out_of_sample(W1,W2,W3)
		ete.append(test_error)
		test_error = in_sample(W1,W2,W3)
		etr.append(test_error)
	epochs = epochs -1

print etr
print ete
print y

plt.plot(y,etr, 'r--',y,ete, 'bs')
plt.show()
		
#print "Test error is ",test_error
# print "W1: ",W3.shape
