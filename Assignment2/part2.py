import re
from porter2stemmer import Porter2Stemmer
from random import shuffle
import math
import numpy as np

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
		sub_enc.append(-1)
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
	if len(weights)!=len(inputs):
		print "error"
		raise Exception
	convolution = weights[-1]
	for i in range(len(weights)-1):
		convolution += weights[i] * inputs[i]
	return tanh_activate(convolution)
 
##### Neuron activation function
def sigmoid_activate(convolution):
	return 1.0 / (1.0 + exp(-convolution))

def tanh_activate(convolution):
	return math.tanh(convolution)

##### Derivative function
def sigmoid_derivative(output):
	return output * (1.0 - output)

def tanh_derivative(output):
	return  (1.0 - (output ** 2))

W1 = np.random.randn(len(tokens)+1, 100).astype(np.float32) * np.sqrt(2.0/(len(tokens)+1))
W2 = np.random.randn(101,50).astype(np.float32) * np.sqrt(2.0/(101))
W3 = np.random.randn(51,2).astype(np.float32) * np.sqrt(2.0/(51))
LR = 0.1
e = 0
for instance in training[:50]:
	x1 = []
	x2 = []
	x3 = []
	# print type(W1)
	instance1 = instance[:len(instance)-1]
	instance1.append(1)
	# print W1.shape
	for i in range(0,100):
		Wtemp = [W1[j][i] for j in range(W1.shape[0])]
		x1.append(forward_propagte(Wtemp,instance1))
	x1.append(1)

	for i in range(0,50):
		Wtemp = [W2[j][i] for j in range(W2.shape[0])]
		x2.append(forward_propagte(Wtemp,x1))
	x2.append(1)

	for i in range(0,2):
		Wtemp = [W3[j][i] for j in range(W3.shape[0])]
		x3.append(forward_propagte(Wtemp,x2))

	z_exp = [math.exp(i) for i in x3]
	sum_z_exp = sum(z_exp)
	softmax = [round(i / sum_z_exp, 3) for i in z_exp]

	h = softmax.index(max(softmax))
	e = 0.5 * ((h - instance[-1]) ** 2)
	print h,instance[-1]
	dell = (h - instance[-1]) * tanh_derivative(h)
	del3 = []
	if h == 0:
		del3.append(softmax[0] * (1 - softmax[1]))
		del3.append(-softmax[0] * softmax[1])
	else:
		del3.append( softmax[1] * (1 - softmax[0]))
		del3.append( -softmax[0] * softmax[1])

	del3 = [(del3[i] * dell) for i in range(len(del3))]
 
	temp3 = sum([W3[i][j] * del3[j] for i in range(W3.shape[0]) for j in range(W3.shape[1]) ])
	del2 = [(((1 - x2[i]) ** 2) * temp3) for i in range(len(x2))]
	temp2 = sum([W2[i][j] * del2[j] for i in range(W2.shape[0]) for j in range(W2.shape[1]) ])
	del1 = [(((1 - x1[i]) ** 2) * temp2) for i in range(len(x1))]

	W3 = [[float(W3[i][j] - LR * x2[i] * del3[j]) for j in range(W3.shape[1])] for i in range(W3.shape[0])]
	W2 = [[float(W2[i][j] - LR * x1[i] * del2[j]) for j in range(W2.shape[1])] for i in range(W2.shape[0])]
	W1 = [[float(W1[i][j] - LR * instance1[i] * del1[j]) for j in range(W1.shape[1])] for i in range(W1.shape[0])]
	W1 = np.asarray([np.asarray(xi) for xi in W1])
	W2 = np.asarray([np.asarray(xi) for xi in W2])
	W3 = np.asarray([np.asarray(xi) for xi in W3])
	# print "W1: ",W3.shape

print "Training done"
print e

test_error = 0.0
for instance in test[:20]:
	x1 = []
	x2 = []
	x3 = []
	instance1 = instance[:len(instance)-1]
	instance1.append(1)
	for i in range(0,100):
		Wtemp = [W1[j][i] for j in range(W1.shape[0])]
		x1.append(forward_propagte(Wtemp,instance1))
	x1.append(1)
	for i in range(0,50):
		Wtemp = [W2[j][i] for j in range(W2.shape[0])]
		x2.append(forward_propagte(Wtemp,x1))
	x2.append(1)
	for i in range(0,2):
		Wtemp = [W3[j][i] for j in range(W3.shape[0])]
		x3.append(forward_propagte(Wtemp,x2))

	z_exp = [math.exp(i) for i in x3]
	sum_z_exp = sum(z_exp)
	softmax = [round(i / sum_z_exp, 3) for i in z_exp]

	h = softmax.index(max(softmax))
	e = 0.5 * ((h - instance[-1]) ** 2)

	print h,instance[-1]
	test_error += e
print "Test done"
print test_error