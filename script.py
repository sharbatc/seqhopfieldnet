import sys
import numpy as np
import hopfield_experiment as exp 
import visualize as vis

# capacity testing

max_iter = 10
num_neurons = [500]
corr = [0]


# for n in num_neurons:
# 	for c in corr:
# 		seq_length = 120
# 		while (exp.avg_generated_sequence(max_iter, seq_length,n,c) > 0.5 ):
# 			print('{} within capacity'.format(seq_length))
# 			seq_length+=1
# 		print('Capacity of {} neurons in a hopfieldnet is {} with corr {}'.format(n,seq_length, c))

# plotting

seq_length = 31
num_neurons = 10
corr = 0
hopfield_dict = exp.read_data(seq_length,num_neurons,corr)
pattern_list = hopfield_dict['pattern_list']
states_as_patterns = hopfield_dict['states_as_patterns']
weight_matrix = hopfield_dict['weight_matrix']
vis.plot_seq_recall(pattern_list,states_as_patterns)
vis.plot_max_overlap_amount(pattern_list,states_as_patterns)

vis.plot_all_overlaps(pattern_list,states_as_patterns,[10,29])
