#Name: Peng Cheng UIN: 674792652
import numpy as np
import h5py
import time
import copy
from random import randint

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )

MNIST_data.close()
####################################################################################
#Implementation of stochastic gradient descent algorithm

#number of inputs
num_inputs = 28*28

#number of outputs
num_outputs = 10

model = {}
model['W1'] = np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)

model_grads = copy.deepcopy(model)
#Include training code here!
######################################################
num_of_itr = 90000
theta = model['W1']
alpha = 0.02

def indicator_vec_generate(y):
	List = []
	for k in range(10):
		if y == k:
			List.append(1)
		else:
			List.append(0)
	return np.array(List)

def softmax(x):
	total = 0
	for i in x:
		total += np.exp(i)
	processed_x = []
	for i in x:
		processed_x.append(np.exp(i)/total)
	return np.array(processed_x)

def gradient(x, y, theta):
	mat1 = indicator_vec_generate(int(y)).reshape(10,1) - softmax(np.dot(theta, x)).reshape(10,1)
	mat2 = x.transpose()
	mat2 = mat2.reshape(1,784)
	grad = (-1) * np.dot(mat1, mat2)
	return grad

def train(trainingX, trainingY, Theta, alpha):
	index_set = np.random.choice(60000,num_of_itr,replace = True) #the size of x_train is 60000
	for l in range(num_of_itr):
		index = index_set[l]
		grad = gradient(trainingX[index], trainingY[index], Theta)
		Theta = Theta - alpha * grad
	return Theta

def forward(x, model):
	final_theta = model["W1"]
	return softmax(np.matmul(final_theta, x))


new_theta = train(x_train, y_train, theta, alpha)
model['W1'] = new_theta
#test data
total_correct = 0
for n in range( len(x_test)):
	y = y_test[n]
	x = x_test[n][:]
	p = forward(x, model)
	prediction = np.argmax(p)
	if (prediction == y):
		total_correct += 1
accuracy = total_correct/np.float(len(x_test) )
print(accuracy)
