import pickle
import plot_tools_seq as plot_tools

def 
file = open(r'data/seqlen3_neunum10_corr0.0.pkl','rb')
new_d = pickle.load(file)
file.close()

pattern_list = new_d['pattern_list']
plot_tools.plot_pattern_list(pattern_list)
