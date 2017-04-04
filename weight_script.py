import pickle
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import visualize as vis 
import hopfield_experiment as exp 

seq_length = 50
num_neurons = 200
corr = 0
iter_no = 5

hopf_dict = exp.read_data(seq_length,num_neurons,corr,iter_no)

w = hopf_dict['weight_matrix']

w_2d = np.array([[0,0]])

for i in np.arange(0,num_neurons,1):
	for j in np.arange(0,i,1):
		w_2d = np.append(w_2d,[[w[i,j],w[j,i]]],axis=0)

w_2d = w_2d[1:]
np.corrcoef(w_2d,rowvar=0)