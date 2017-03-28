import numpy as np
import matplotlib.pyplot as plt

import hopfield_network_seq as network
import pattern_tools_seq as pattern_tools
import plot_tools_seq as plot_tools


np.random.seed(314)

#### create letters start ###

letter_list = ['I','J','L','F','P','E']

abc_dictionary =pattern_tools.load_alphabet()

# access the first element and get it's size (they are all of same size)
pattern_shape = abc_dictionary['A'].shape
print("letters are patterns of size: {}. Create a network of corresponding size".format(pattern_shape))

# create a list using Pythons List Comprehension syntax:
pattern_list = [abc_dictionary[key] for key in letter_list ]
plot_tools.plot_pattern_list(pattern_list)
hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_shape[0]*pattern_shape[1])


#### create letters end ###

### create correlated patterns ###


pattern_size = 10

factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
l_pattern = factory.create_Lnew_pattern(0)
pattern_list = [l_pattern]
for i in np.arange(1,6,1):
	pattern_list.append(factory.create_Lnew_pattern(i))

plot_tools.plot_pattern_list(pattern_list)
hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_size**2)

### create correlated patterns end ###



# store the patterns
hopfield_net.store_patterns(pattern_list)

# # create a noisy version of a pattern and use that to initialize the network

noisy_init_state = pattern_tools.get_noisy_copy(abc_dictionary['A'], noise_level=0.2)
# noisy_init_state = pattern_tools.get_noisy_copy(factory.create_Lnew_pattern(0), noise_level=0)

hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=15)

plot_tools.plot_network_weights(hopfield_net)
# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = pattern_tools.reshape_patterns(states, pattern_list[0].shape)

# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, None, suptitle="Network dynamics")




