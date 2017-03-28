import sys
import numpy as np
import hopfield_experiment as exp 
import visualize as v

# capacity testing

max_iter = 5
num_neurons_rt= 150
seq_length = 0
int(0.01*(num_neurons_rt*num_neurons_rt))
corr = 0.8


while (exp.avg_generated_sequence(max_iter, seq_length,num_neurons_rt, corr) > 0.5 ):
	print('{} within capacity'.format(seq_length))
	seq_length+=1

print('Capacity of {} neurons in a hopfieldnet is {}'.format(num_neurons_rt,seq_length))

