############################
#
# Name: Kevin Nhan
# ID: 0971282
# Assignment 1
# CSCI 4391 - Intro to Machine Learning
#
# Description: 
# The following program generates a 20 x 3 matrix with 
# random integers from 0 to 20 for the two features and 
# -1 or 1 for the classification. Using this lineraly seperable 
# data, the program uses the PLA algorithm to find the ideal
# weights to correctly classify this data. A scatter plot of the
# data and the linear seperator is then generated to show the
# results. 
#
############################

import numpy as np
import matplotlib.pyplot as plt 
import random 

def randomArray(x, start, stop):
	randArray = []
	for i in range(x):
		x = random.randint(start, stop)
		randArray.append(x)
	return randArray 

#Classifies data and adds to dataset 
def createData(target, x1, x2, classify):
	for i in range(len(x1)):
		temp = []
		temp.append(x1[i])
		temp.append(x2[i])
		temp.append(classify)
		target.append(temp)

#sign(h(x)) for PLA. d = dataset/array, w = weight vector
def h(d, w, row_index, feature_size):
	wT = w.T #transpose weight vector
	x_initial = np.array([d[row_index, 0:feature_size]]) #row vector of x with only features
	x_add_x0 = np.array([np.insert(x_initial, 0, 1)]) #insert x0 = 1
	x = x_add_x0.T #row vector of x transposed
	product = wT @ x
	result = np.sign(product) #sign of wTx
	if result >= 0: #0 is treated as positive
		return 1
	else:
		return -1

#Takes weight vector as argument
def update(misclass, feature_size):
	x_initial = misclass[0:feature_size] #x vector without x0 = 1
	x_add_x0 = np.insert(x_initial, 0, 1) #add x0 = 1
	x_vector = np.array([x_add_x0]) #create x row vector
	x = x_vector.T #transpose
	y = int(misclass[feature_size:]) #correct classification value
	return y * x

#First group
x1_pts = randomArray(10, 0, 10)
y1_pts = randomArray(10, 0, 10)
#Second group
x2_pts = randomArray(10, 12, 20)
y2_pts = randomArray(10, 12, 20)
#Classify groups and combine into dataset named matrix
matrix = []
createData(matrix, x1_pts, y1_pts, -1)
createData(matrix, x2_pts, y2_pts, 1)

a = np.array(matrix)
print("Dataset: ")
print(a) #Print dataset
a_dim = a.shape
row_size = a_dim[0]

w = np.zeros((3, 1), dtype=np.int16) #intial weight vector set to 0
num_of_misclass = 1 #default value to start while loop
#PLA Loop 
while num_of_misclass != 0:
	misclassified = []
	#Loop for h(x)
	for i in range(row_size):
	 	y1 = h(a, w, i, 2)
	 	if y1 != int(a[i, 2:]):
	 		misclassified.append(a[i])
	#Misclassification & Weight update
	num_of_misclass = len(misclassified)
	if num_of_misclass > 0:
		choice = random.randint(0, num_of_misclass - 1) #pick random misclassification
		misclass_vector = misclassified[choice]
		w += update(misclass_vector, 2) #update weight vector

#Print weight vector
print("Weight Vector: ")
print(w)

#Plot linear seperator
w0 = int(w[0])
w1 = int(w[1])
w2 = int(w[2])
if w2 != 0:
        slope = (w1/w2) * -1 #multiply by -1 to flip sign
        intercept = (w0/w2) * -1
        x = np.linspace(0,20,100)
        y = slope*x + intercept
        plt.plot(x, y)
else: #In case w2 is 0, so line is x = intercept
        intercept = (w0/w1) * -1
        plt.axvline(x=intercept)

#Plot data points
area = np.pi*3
plt.scatter(x1_pts, y1_pts, s=area, c='red', alpha=0.5)
plt.scatter(x2_pts, y2_pts, s=area, c='blue', alpha=0.5)
#Display plot
plt.show()











