import sys
import numpy as np
import hopfield_experiment as exp 
import visualize as v

# capacity testing

max_iter = 5
neuro_root = 10
seq_length = 0.1*(neuro_root*neuro_root)
corr = 0

while (exp.avg_generated_sequence(max_iter, seq_length,neuro_root, corr) > 0.9 ):
	print('{} within capacity'.format(seq_length))
	seq_length+=1

