# pong_rnn.py
# Dave Gottlieb 2016 (dmg1@stanford.edu)
# ======
# 
# Recurrent network built using Keras. 
# 
# Intended behavior: learn the state transitions of a Pong game engine using a large amount of Pong data. 
#
# Given: nothing except screen buffers and input streams. 

from keras.models import Graph
from keras.layers.core import TimeDistributedDense, Flatten, Reshape, Dense, Activation 
from keras.layers.recurrent import SimpleRNN
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback


class PongRNN(object): 

	def __init__(self):
		self.graph = Graph()

		reg = l2(0.001)
		CONV3_1 = Convolution2D(nb_filter=8, nb_row=3, nb_col=3, init='he_uniform', activation='relu', border_mode='same', W_regularizer=reg)
		CONV3_2 = Convolution2D(nb_filter=16, nb_row=3, nb_col=3, init='he_uniform', activation='relu', border_mode='same', W_regularizer=reg)
		RNN3 = SimpleRNN(output_dim=512, input_dim=16386, input_length=9)
		FC3 = Dense(1024, activation='relu', init='he_uniform')
		FC4 = Dense(1024)

		self.graph.add_input(name='screen_in', input_shape=(1,32,32)) 
		self.graph.add_node(CONV3_1, name='CONV3_1', input='screen_in')
		self.graph.add_node(CONV3_2, name='CONV3_2', input='CONV3_1')
		self.graph.add_node(Flatten(), name='FLATTEN', input='CONV3_2')

		self.graph.add_input(name='control_in', input_shape=(2,))
		self.graph.add_node(Reshape((-1,2)), name='RESHAPE3', input='control_in')

		self.graph.add_node(Reshape((-1, 16384)), name='RESHAPE', input='FLATTEN')


		self.graph.add_node(RNN3, name='RNN3', inputs=['RESHAPE', 'RESHAPE3'])

		#self.graph.add_node(Reshape((-1, 256)), name='RESHAPE2', input='RNN3')

		self.graph.add_node(FC3, name='FC3', input='RNN3')

		self.graph.add_node(FC4, name='FC4', input='FC3')

		self.graph.add_node(Activation('softmax'), name='SOFTMAX', input='FC4')

		#self.graph.

		self.graph.add_output(name='screen_out', input='SOFTMAX')

		self.graph.compile(optimizer='adam', loss={'screen_out': 'binary_crossentropy'})

	def train(self, q_train, p_train, y_train, nb_epoch=1, batch_size=128, checkpoints=False): 
		
		callbacks = []

		if checkpoints: 
			checkpointer = ModelCheckpoint(filepath="~/weights/weights.hdf5", verbose=1, save_best_only=True)
			callbacks.append(checkpointer)
			class S3Uploader(keras.callbacks.Callback):
				def on_epoch_end(self,epoch, logs={}): 
					import subprocess
					subprocess.call("aws s3 sync ~/weights s3://model-checkpoints", shell=True)
			uploader = S3Uploader()
			callbacks.append(uploader)


		self.history = self.graph.fit({'screen_in': q_train, 'control_in': p_train, 'screen_out':y_train}, nb_epoch=nb_epoch, callbacks=callbacks)#, batch_size=batch_size)
		# Todo: write a method that calls pong as a generator, rather than using pre-generated data. 

	def test(self, q, p, y):
		print self.graph.test_on_batch({'screen_in': q, 'control_in': p, 'screen_out': y}, accuracy=True)

	def predict(self, q, p, batch_size=128, verbose=0):
		return self.graph.predict({'screen_in': q, 'control_in': p}, batch_size=batch_size, verbose=verbose)


	def loss(self, y_true, y_pred):
		pass 
		# for now I use mse or sim. this is the wrong objective function but the right one needs to be specifically engineered 
		# and written as a Theano symbolic function 