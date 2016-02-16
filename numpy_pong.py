# numpy_pong.py
# Dave Gottlieb 2016 (dmg1@stanford.edu)
# =====
#
# small numpy implementation of Pong for programmatic interaction
#
# matplotlib is used for live visualization (in __main__)

import numpy as np
import matplotlib.pyplot as plt
import time


class PongState(object):

	

	def __init__(self):
		self.screen_buffer = np.zeros((32,32), dtype=np.uint8)
		self.paddle_1 = self.Paddle(side='left')
		self.paddle_2 = self.Paddle(side='right')
		self.ball = self.Ball()
		self.scores = None


	def draw(self):
		self.screen_buffer = np.zeros((32,32), dtype=np.uint8)

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
		pass

	def update_ball(self, ball, paddle_1, paddle_2): 
		
		# collision detection
		for p in [paddle_1, paddle_2]: 
			p_closest = np.argmin(np.absolute(np.arange(p.top[0], p.top[0]+p.size) - ball.top[0]+ball.size/2))
			d = ball.top - np.asarray((p_closest, p.top[1]))
			if np.linalg.norm(d) < 2: 
				ball.momentum[1] *= -1
				ball.momentum[0] += p.momentum[0]

		# decay / clip momentum
		ball.momentum = np.clip(ball.momentum, -2, 2) 


		# position update
		ball.top += ball.momentum

		# check for out-of-bounds
		if (ball.top[1] < 0 or ball.top[1] > 31): 
			ball = self.Ball(top=(15,15), size=2)

			# score logic would go here

	class Ball(object):
		
		def __init__(self, top=(15,15), size=2):
			self.top = top
			self.size=size
			self.momentum = (1,1)

	class Paddle(object): 
		
		def __init__(self, side='left', size=5): 
			self.top = (13,2)
			if side == 'right': 
				self.top = (13, 29)

			self.size = size
			self.momentum = 0 






def main():
	input1 = None
	input2 = None
	pong = PongState()
	keep_going = True

	plt.axis([0,32,0,32])
	#plt.ion()
	#plt.show()

	while(keep_going): 
		pong.update(input1, input2)
		screen = pong.draw()
		plt.imshow(screen)
		plt.draw()
		plt.show()
		time.sleep(5.0)
		#keep_going = False
		#print np.where(screen != 0)


if __name__ == "__main__":
	main()

