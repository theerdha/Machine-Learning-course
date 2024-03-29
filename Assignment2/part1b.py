import re
from porter2stemmer import Porter2Stemmer
from random import shuffle
import math
import numpy as np
import matplotlib.pyplot as plt

stoplist = []
f = open("/home/sree/Machine-Learning-course/Assignment2/stopwords.txt",'r')
for line in f: 
	stoplist.append(line.replace("\n",""))
#print stoplist

stemmer = Porter2Stemmer()


# In[5]:


wosw = []
f = open("/home/sree/Machine-Learning-course/Assignment2/Assignment_2_data.txt",'r')
for line in f: 
	z = re.split('\t| |;|,|\*|\n|\.',line)
	b = filter(lambda a: a != '', z)
	l3 = [stemmer.stem(x) for x in b if x.lower() not in stoplist]
	wosw.append(l3)
shuffle(wosw)


# In[6]:


tokens = []
for row in wosw:
    for item in row:
        if item != 'ham' and item != 'spam':
        	tokens.append(item)
tokens = sorted(set(tokens))
tokens = tokens[:2000]


# In[7]:


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


# In[8]:


len_part=int(math.ceil(len(encoding)*0.8))
training = encoding[:len_part]
test = encoding[len_part:]


# In[9]:


def forward_propagte(weights, inputs):
	return tanh_activate(np.dot(inputs,weights))
 
##### Neuron activation function
def sigmoid_activate(convolution):
	return 1.0 / (1.0 + exp(-convolution))

def tanh_activate(convolution):
	return np.tanh(convolution)

##### Derivative function
def sigmoid_derivative(output):
	return output * (1.0 - output)

def tanh_derivative(output):
	return  (1.0 - (output ** 2))


# In[10]:


def out_of_sample(W1,W2,W3):
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
	h = forward_propagte(W3,x2)
	x = np.ones((h.shape[0],h.shape[1]),dtype=int)
	testxx = testx[:,-1]
	for count in range(0,h.shape[0]): 
		if h[count][0] > 0:
			x[count][0] = 1
		else:
			x[count][0] = -1
		if x[count][0] != testxx[count] :
	 		misclass = misclass + 1
		test_error = test_error + 0.5 * ((h[count][0] - test[count][-1]) ** 2)

	#print "out of sample misclass " ,misclass
	#print "out of sample test error",test_error 
	return [misclass,test_error] 


# In[11]:


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
	h = forward_propagte(W3,x2)
	x = np.ones((h.shape[0],h.shape[1]),dtype=int)
	testxx = testx[:,-1]

	for count in range(0,h.shape[0]): 
		if h[count][0] > 0:
			x[count][0] = 1
		else:
			x[count][0] = -1

		if x[count][0] != testxx[count] :
	 		misclass = misclass + 1
		test_error = test_error + 0.5 * ((h[count][0] - training[count][-1]) ** 2)

	#print "insample misclass " ,misclass
	#print "insample test error",test_error 
	return [misclass,test_error]


# In[12]:


W1 = np.random.normal(0,1,(len(tokens)+1, 100)).astype(np.float32) * np.sqrt(2.0/(len(tokens)+1))
W2 = np.random.normal(0,1,(101,50)).astype(np.float32) * np.sqrt(2.0/(101))
W3 = np.random.normal(0,1,(51,1)).astype(np.float32) * np.sqrt(2.0/(51))
LR = 0.1
etr = []
ete = []
y = []

print("Training started")

epochs = 1
while epochs < 1000:
	shuffle(training)
	print "epoch : ",epochs
	for instance in training:
		# print type(W1)
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
		
		del3 = np.asscalar((h[0] - instance[-1]) * tanh_derivative(h[0]))

		temp3 = del3*W3.T
		del2 = (1-np.square(x2)) * temp3
		
		temp2 = np.dot(W2,del2[:,0:del2.shape[1]-1].T)
		del1 = (1-np.square(x1)) * temp2.T


		W3 = W3 - LR * del3 * x2.T
		W2 = W2 - LR * np.dot(x1.T,del2[:,0:del2.shape[1]-1])
		W1 = W1 - LR * np.dot(instance1.T,del1[:,0:del1.shape[1]-1])
	if epochs % 100 == 0:
		[misclass,test_error] = out_of_sample(W1,W2,W3)
		print "out sample error ", test_error , " out sample misclass ", misclass,"/",len(test)
		ete.append(test_error)
		[misclass,test_error] = in_sample(W1,W2,W3)
		print "in sample error ", test_error , " in sample misclass ", misclass,"/",len(training)
		etr.append(test_error)
		y.append(epochs)
	
	if test_error < 60:
		break
	epochs = epochs +1

print "training error"
print etr
print "test error"
print ete
print "epochs"
print y

plt.plot(y,etr,'-',label='Insample Error')
plt.plot(y,ete,'-',label='Outsample Error')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error vs Epochs in part 1b")
plt.legend()
plt.show()
