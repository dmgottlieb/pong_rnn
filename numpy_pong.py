# numpy_pong.py
# Dave Gottlieb 2016 (dmg1@stanford.edu)
# =====
#
# small numpy implementation of Pong for programmatic interaction
#
# matplotlib is used for live visualization (in __main__)
#
# game can be started with a seed to get deterministic outcomes

import numpy as np


class PongState(object):

	

	def __init__(self, seed=None):
		self.rnd = np.random.RandomState(seed)
		self.screen_buffer = np.zeros((32,32), dtype=np.int8)
		self.paddle_1 = self.Paddle(side='left', rnd=self.rnd)
		self.paddle_2 = self.Paddle(side='right', rnd=self.rnd)
		self.ball = self.Ball(rnd=self.rnd)
		self.scores = None


	def draw(self):
		self.screen_buffer = np.zeros((32,32), dtype=np.int8)

		self.draw_scores(self.screen_buffer, self.scores)
		self.draw_paddle(self.screen_buffer, self.paddle_1)
		self.draw_paddle(self.screen_buffer, self.paddle_2)
		self.draw_ball(self.screen_buffer, self.ball)
		
		return self.screen_buffer

	def update(self, input1, input2):
		self.update_paddle(self.paddle_1, input1)
		self.update_paddle(self.paddle_2, input2)

		self.update_ball(self.ball, self.paddle_1, self.paddle_2)

	def draw_scores(self, screen_buffer, scores):
		pass

	def draw_paddle(self, screen_buffer, paddle): 
		top, size = paddle.top, paddle.size
		screen_buffer[top[0]:top[0]+size, top[1]] = 1

	def draw_ball(self, screen_buffer, ball): 
		top, size = ball.top, ball.size
		screen_buffer[top[0]:top[0]+size, top[1]:top[1]+size] = 1

	def update_paddle(self, paddle, input): 
		# detect input
		# in this model, input is simply {-1, 1} corresponding to moving down or up respectively
		# 0 is no movement

		if input == None:
			dm = 0
		else: 
			dm = input 


		# update momentum
		paddle.momentum += dm

		# decay / clip momentum
		paddle.momentum = np.clip(paddle.momentum, -2,2)

		# check for out-of-bounds
		if (paddle.top[0] < 0 or paddle.top[0] + paddle.size > 31): 
			pass#paddle.momentum *= -1

		# update position
		paddle.top[0] += paddle.momentum
		paddle.top[0] = np.clip(paddle.top[0], 0, 31 - paddle.size)



	def update_ball(self, ball, paddle_1, paddle_2): 
		
		# collision detection
		# case: paddles
		for p in [paddle_1, paddle_2]: 
			p_closest = np.argmin(np.absolute(np.arange(p.top[0], p.top[0]+p.size) - ball.top[0]+ball.size/2)) + p.top[0]
			d = ball.top - np.asarray((p_closest, p.top[1]))
			if np.linalg.norm(d) < 2: 
				ball.momentum[1] *= -1
				ball.momentum[0] += p.momentum

		# case: top and bottom edges
		if (ball.top[0] < 0 or ball.top[0] > 32):
			ball.momentum[0] *= -1


		# decay / clip momentum
		ball.momentum = np.clip(ball.momentum, -3, 3) 


		# position update
		ball.top += ball.momentum

		# check for out-of-bounds
		if (ball.top[1] < 0 or ball.top[1] > 31): 
			ball.__init__()

			# score logic would go here

	class Ball(object):
		
		def __init__(self, top=(15,15), size=2, rnd=np.random.RandomState()):
			self.rnd = rnd
			self.top = np.array(top)
			self.size=size
			m_x = self.rnd.choice([-1,1])
			m_y = self.rnd.choice([-1,1])
			# m_x, m_y = -1,1
			self.momentum = np.array([m_x,m_y])

	class Paddle(object): 
		
		def __init__(self, side='left', size=5, rnd=np.random.RandomState()): 
			self.rnd = rnd
			height = self.rnd.choice(24)
			self.top = np.array((height+4,2))
			if side == 'right': 
				self.top = np.array((height+4, 29))

			self.size = size
			self.momentum = 0 


class BallChaseAI(object): 

	def __init__(self):
		pass

	def getMove(self, ball, paddle): 
		input = 0
		if (ball.top[0] + ball.size/2) > (paddle.top[0] + paddle.size/2): 
			input = 1

		if (ball.top[0] + ball.size/2) < (paddle.top[0] + paddle.size/2): 
			input = -1


		return input



def pyplot_live_mode():
	import matplotlib.pyplot as plt
	import time

	player1 = BallChaseAI()
	player2 = BallChaseAI()
	pong = PongState()
	keep_going = True

	plt.axis([0,32,0,32])
	plt.ion()
	#plt.show()


	i=0
	while(i < 50): 
		input1 = player1.getMove(pong.ball, pong.paddle_1)
		input2 = player2.getMove(pong.ball, pong.paddle_2)
		pong.update(input1, input2)
		screen = pong.draw()
		plt.imshow(screen)
		plt.draw()
		plt.show()
		plt.pause(0.01)
		i+=1
		#keep_going = False
		#print np.where(screen != 0)

def gif_mode():
	from PIL import Image, ImageSequence
	from images2gif import writeGif
	player1 = BallChaseAI()
	player2 = BallChaseAI()
	pong = PongState()
	frames = []


	i=0
	while(i < 768): 
		input1 = player1.getMove(pong.ball, pong.paddle_1)
		input2 = player2.getMove(pong.ball, pong.paddle_2)
		pong.update(input1, input2)
		screen = pong.draw()
		i+=1
		#keep_going = False
		#print np.where(screen != 0)
		# screen = screen[:,:]
		frames.append(Image.fromarray(screen*255, mode='L'))
		# im.show()
		# im.save('frame' + str(i) + '.gif')

	writeGif("frames.gif", frames, duration=0.1)
	# ims = [Image.fromarray(s) for s in screen]
	# ims.save('ani.gif', save_all=True)

def headless_mode(): 
	# returns frames, a list of 1D numpy arrays of shape (32x32 + 2), 
	# containing the screen state at time t (unraveled) followed by the input at time (t-1)
	player1 = BallChaseAI()
	player2 = BallChaseAI()
	pong = PongState()
	frames = []

	i = 0
	input1 = 0
	input2 = 0

	while(i < 100):
		input1 = player1.getMove(pong.ball, pong.paddle_1)
		input2 = player2.getMove(pong.ball, pong.paddle_2)
		pong.update(input1, input2)
		screen = pong.draw().flatten()
		screen = np.append(screen, (input1,input2))
		i+=1
		frames.append(screen)

	# insert ending frame
	frames.append(np.ones_like(screen) * -1)
	return np.array(frames, dtype=np.int8)

def main():
	# from pandas import HDFStore	
	# from pandas import DataFrame as df 
	# import tables

	# store = HDFStore('data.h5')

	for j in range(1):
		# frames = df(headless_mode())
		#store['game_' + str(j)] = frames
		frames = headless_mode()
		np.savetxt('gametest.txt', frames, fmt='%i')

	# store.close()


if __name__ == "__main__":
	main()



