import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.
from pdb import set_trace
style.use("ggplot")


class Blob:
	def __init__(self, size):
		self.size = size
		self.x = np.random.randint(0, self.size)
		self.y= np.random.randint(0,  self.size)
	
	def __str__(self):
		return f"Blob {self.x}, {self.y}"

	def __sub__(self, other):
		return (self.x-other.x, self.y-other.y)

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

	def action(self, choice):
		if choice == 0:
			self.move(x=1, y=0)
		elif choice == 1:
			self.move(x=-1, y=0)
		elif choice == 2:
			self.move(x=0, y=1)
		elif choice == 3:
			self.move(x=0, y=-1)



	def move(self, x=False, y=False):
		if x is False:
			self.x += np.random.randint(-1,2)
		else:
			self.x += x

		if y is False:
			self.y += np.random.randint(-1,2)
		else:
			self.y += y

		if self.x<0:
			self.x =0
		elif self.x > self.size-1:
			self.x = self.size -1

		if self.y<0:
			self.y =0
		elif self.y > self.size-1:
			self.y = self.size -1			

class BlobEnv:
	SIZE = 10
	RETURN_IMAGES = False
	MOVE_PENALTY = 1
	ENEMY_PENALTY = 300
	FOOD_REWARD = 300
	#OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3) ##bgr
	OBSERVATION_SPACE_VALUES = (4,)
	
	ACTION_SPACE_SIZE = 4
	PLAYER_N = 1
	FOOD_N = 2
	ENEMY_N = 3

	d = {1: (255, 175, 0),
		 2: (0, 255, 0),
		 3: (0, 0, 255)}

	def reset(self):
		self.player = Blob(self.SIZE)
		self.food = Blob(self.SIZE)
		self.enemy = Blob(self.SIZE)
		'''
		self.food.x = 2
		self.food.y = 2
		self.enemy.x = 3
		self.enemy.y = 1
		while self.enemy == self.player or self.player == self.food:
			self.player = Blob(self.SIZE)
		'''
		while self.food == self.player:
			self.food = Blob(self.SIZE)

		while self.enemy == self.player or self.enemy == self.food:
			self.enemy = Blob(self.SIZE)
	
		self.episode_step = 0

		if self.RETURN_IMAGES:
			observation = np.array(self.get_image()) / 255
		else:
			observation = (self.player-self.food) + (self.player-self.enemy)
			observation = np.array(observation) / self.SIZE
		return observation

	def step(self, action):
		self.episode_step += 1
		self.player.action(action)
		#### MAYBE ###
		#enemy.move()
		#food.move()
		##############
		if self.RETURN_IMAGES:
			new_observation = np.array(self.get_image()) /255
		else:
			new_observation = (self.player-self.food) + (self.player-self.enemy)
			new_observation = np.array(new_observation) / self.SIZE

		if self.player == self.enemy:
			reward = -self.ENEMY_PENALTY
			done = True
		elif self.player == self.food:
			reward = self.FOOD_REWARD
			done = True
		else:
			reward = -self.MOVE_PENALTY		
			done = False
		#done = False
		#if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY:
		#	done = True

		return new_observation, reward, done, None

	def render(self):
		img = self.get_image()
		img = np.kron(img, np.ones((30, 30, 1)) )
		cv2.imshow("image", img)  # show it!
		cv2.waitKey(1)

	def get_image(self):
		env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
		env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
		env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
		env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
		return env

'''
env = BlobEnv()
ob = env.reset()
print(env.player.x, env.player.y)
#print(ob)
env.render()
ob, r, _, _ = env.step(4)
#print(ob, "  ", r)
print(env.player.x, env.player.y)

env.render()
'''


'''
if __name__ == "__main__":

	SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.998

SHOW_EVERY = 3000

start_q_table = None # or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PIAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1:(255,175,0),
     2:(0,255,0),
     3:(0,0,255) }


	if start_q_table is None:
		q_table = {}
		for x1 in range(-SIZE+1, SIZE):
			for y1 in range(-SIZE+1, SIZE):
				for x2 in range(-SIZE+1, SIZE):
					for y2 in range(-SIZE+1, SIZE):
						q_table[((x1,y1),(x2,y2))] = [np.random.uniform(-5,0) for i in range(4)]
	else:
		 with open(start_q_table, "rb") as f:
		 	q_table = pickle.load(f) 

	episode_rewards = []
	for episode in range(HM_EPISODES):
		player = Blob()
		food = Blob()
		enemy = Blob()

		if episode%SHOW_EVERY == 0:
			print(f"on #{episode}, epsilon is {epsilon}")
			print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
			show = True
		else:
			show = False

		episode_reward = 0
		for i in range(200):
			obs = (player-food, player-enemy)
			if np.random.random() > epsilon:
				action = action = np.argmax(q_table[obs])
			else:
				action = np.random.randint(0, 4)
			player.action(action)

			if player.x == enemy.x and player.y == enemy.y:
				rewrd = -ENEMY_PENALTY
			elif player.x == food.x and player.y == food.y:
				reward = FOOD_REWARD
			else:
				reward = -MOVE_PENALTY
			
			new_obs = (player-food, player-enemy)  # new observation
			max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
			current_q = q_table[obs][action]  # current Q for our chosen action
'''



