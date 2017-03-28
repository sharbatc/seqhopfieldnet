import sys
import numpy as np
import hopfield_experiment as exp 
import visualize as v

# capacity testing

max_iter = 5
max_seq_length = 30
neuro_root = 10
corr = 0

while (exp.avg_generation(max_iter, max_seq_length,neuro_root, corr) > 0.9 ):
	print('{} within capacity'.format(max_seq_length))
	max_seq_length+=1

