# hidden_state_viz.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
# 
# To convince myself that the recurrency is working right, and to see if there are any interesting patterns, 
# a tool for extracting the hidden state activations from Pong RNN. They can then be plotted over time. 

import theano

def get_hidden_f(control, screen, RNN):
	"""
	takes as arguments the model inputs and the layer whose 
	activation we are trying to visualize. 

	Returns a Theano symbolic function that will calculate activations
	at the desired layer given inputs.
	"""
	hidden_f = theano.function([screen.get_input(train=False), 
		control.get_input(train=False)],
		RNN.get_output(train=False))
	return hidden_f
