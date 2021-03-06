'''
Sharbatc
EPFL, Lausanne, March 2017.
Experiment iterating over the parameters.
Defines functions where one creates data files to be stored later for visualization.
'''
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hopfield_network_seq as network
import pattern_tools_seq as pattern_tools
import plot_tools_seq as plot_tools

def generate_sequence(seq_length,num_neurons,corr,corr_tolerance = 0.005):
	'''
	Generates sequences to be fed to the Hopfield net
	'''
	factory = pattern_tools.PatternFactory(num_neurons, 1)
	pattern_n = factory.create_random_pattern()

	pattern_list = [pattern_n] #start the seq. min 1 to make sense

	## create the sequences ##
	for num in np.arange(1, seq_length,1):
		pattern_nplus1 = pattern_tools.get_noisy_copy(pattern_n, 1-corr)
		corr_diff = np.abs(np.corrcoef(pattern_n.flatten(),pattern_nplus1.flatten())[0,1]) - corr # how far are we from desired corr?
		
		while corr_diff > corr_tolerance: #keep generating noisy ones ad okay-um
		        pattern_nplus1 = pattern_tools.get_noisy_copy(pattern_n, 1-corr)
		        corr_diff = np.abs(np.abs((np.corrcoef(pattern_n.flatten(),pattern_nplus1.flatten())[0,1])) - corr)

		pattern_list.append(pattern_nplus1);
		pattern_n = pattern_nplus1;

	return pattern_list

def run_seq_hopfield_net(seq_length,num_neurons,corr,iter_no,save_flag=1):
	'''
	Function to be called each time for the experiment to run the hopfield network. Stores the states as a pattern returned by the hopfield network as well the pattern stored in a dictionary and dumps it into .pkl file to be used later for further analysis.

	'''
	pattern_list = generate_sequence(seq_length,num_neurons,corr)
	hopfield_net = network.HopfieldNetwork(nr_neurons=num_neurons);
	hopfield_net.store_patterns(pattern_list);

	#initialise with initial pattern
	noisy_init_state = pattern_tools.get_noisy_copy(pattern_list[0],noise_level=0); #exactly get the first pattern
	hopfield_net.set_state_from_pattern(noisy_init_state);
	states = hopfield_net.run_with_monitoring(nr_steps=seq_length+1);
	states_as_patterns = pattern_tools.reshape_patterns(states, pattern_list[0].shape);

	dictionary = {'pattern_list':pattern_list,'states_as_patterns':states_as_patterns,'weight_matrix':hopfield_net.weights}
	## save data ##
	if save_flag:
		file_name = 'data/seqlen{}_neunum{}_corr{}_iter_no{}.pkl'.format(seq_length,num_neurons,corr,iter_no)
		# print(file_name)
		afile = open(file_name,'wb')
		pickle.dump(dictionary,afile)
		afile.close()
	# else : 
	# 	return dictionary

def read_data(seq_length,num_neurons,corr,iter_no):
	file_name = 'data/seqlen{}_neunum{}_corr{}_iter_no{}.pkl'.format(seq_length,num_neurons,corr,iter_no)
	file = open(file_name,'rb')
	new_d = pickle.load(file)
	file.close()
	return new_d

def is_seq_generated(pattern_list,states_as_patterns):
	'''
	Computes if sequence is generated or not

	'''
	seq_length=len(pattern_list)
	seq_gen = np.array([])
	overlap_gen = np.array([])
	for i in range(len(states_as_patterns)):
		overlap_list = pattern_tools.compute_overlap_list(states_as_patterns[i], pattern_list)
		# print(overlap_list.argmax(),overlap_list.max())
		seq_gen = np.append(seq_gen,overlap_list.argmax())
		overlap_gen = np.append(overlap_gen,overlap_list.max())

	# print(seq_gen)
	# print(overlap_gen)
	
	if np.size(np.nonzero(seq_gen[:seq_length] - np.arange(0,seq_length,1))):
		return 0 #not generated
	else :
		return 1 #generated


def avg_generated_sequence(max_iter, seq_length,num_neurons, corr,save_flag=1):
	output = np.zeros(max_iter)
	for i in np.arange(max_iter):
		run_seq_hopfield_net(seq_length,num_neurons,corr,i)
		hopfield_dict = read_data(seq_length,num_neurons,corr,i)
		pattern_list = hopfield_dict['pattern_list']
		states_as_patterns = hopfield_dict['states_as_patterns']
		output[i] = is_seq_generated(pattern_list,states_as_patterns)
	return np.mean(output)

if __name__ == "__main__":
	import sys
	import time
	start_time = time.time()
	seq_length = int(sys.argv[1])
	num_neurons_rt = int(sys.argv[2])
	corr = float(sys.argv[3])

	run_seq_hopfield_net(seq_length,num_neurons_rt,corr,1)

	hopfield_dict = read_data(seq_length,num_neurons_rt,corr,1)

	pattern_list = hopfield_dict['pattern_list']
	states_as_patterns = hopfield_dict['states_as_patterns']
	if (is_seq_generated(pattern_list,states_as_patterns)):
		print('sequence generated')
	else:
		print('not generated, beyond capacity')


