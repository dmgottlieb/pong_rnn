# data_generation.py
# Dave Gottlieb 2016 (dmg1@stanford.edu)



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



