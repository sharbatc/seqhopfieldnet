import pickle
import plot_tools_seq as plot_tools
import pattern_tools_seq as pattern_tools
import matplotlib.pyplot as plt
import numpy as np

def plot_seq_recall(pattern_list, states_as_patterns):
	seq_length=len(pattern_list)
	seq_gen = np.array([])
	overlap_gen = np.array([])
	for i in range(len(states_as_patterns)):
		overlap_list = pattern_tools.compute_overlap_list(states_as_patterns[i], pattern_list)
		# print(overlap_list.argmax(),overlap_list.max())
		seq_gen = np.append(seq_gen,overlap_list.argmax())
		overlap_gen = np.append(overlap_gen,overlap_list.max())

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim(-0.1,seq_length+2)
	ax.set_ylim(-0.1,seq_length+2)
	ax.grid(True)
	seq_gen, = ax.plot(np.arange(0,seq_length+2,1),seq_gen,'o')
	ax.set_xlabel('State Evolution')
	ax.set_ylabel('Sequence Generated')

	plt.tight_layout
	plt.show()

def plot_max_overlap_amount(pattern_list, states_as_patterns):	
	seq_length=len(pattern_list)
	seq_gen = np.array([])
	overlap_gen = np.array([])
	for i in range(len(states_as_patterns)):
		overlap_list = pattern_tools.compute_overlap_list(states_as_patterns[i], pattern_list)
		# print(overlap_list.argmax(),overlap_list.max())
		seq_gen = np.append(seq_gen,overlap_list.argmax())
		overlap_gen = np.append(overlap_gen,overlap_list.max())

	fig = plt.figure()
	ax2 = fig.add_subplot(111)
	ax2.bar(range(len(overlap_gen)),overlap_gen)
	ax2.grid(True)
	ax2.set_xlabel('State Evolution')
	ax2.set_ylabel('Maximum Overlap Amount')

	plt.tight_layout
	plt.show()

def plot_all_overlaps(pattern_list, states_as_patterns):
	seq_length=len(pattern_list)
	seq_gen = np.array([])
	overlap_gen = np.array([])
	for i in range(len(states_as_patterns)):
		overlap_list = pattern_tools.compute_overlap_list(states_as_patterns[i], pattern_list)
