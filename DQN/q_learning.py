import gym
import numpy as np 
import matplotlib.pyplot as plt
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 4000

SHOW_EVERY = 500
STATS_EVERY = 100

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

env = gym.make("MountainCar-v0")
env.reset()

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE+[env.action_space.n]))

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING  = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

ep_rewards=[]
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}


for episode in range(EPISODES):
	episode_reward = 0
	if episode % SHOW_EVERY ==0:
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset()) 
	done = False
	while not done:
		action = np.argmax(q_table[discrete_state])
		new_state, reward, done, _ = env.step(action)
		episode_reward += reward
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state+(action,)]
			new_q = (1 - LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
			q_table[discrete_state+(action,)] = new_q
		
		discrete_state = new_discrete_state

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

	ep_rewards.append(episode_reward)
	if not episode % STATS_EVERY:
		average_reward = sum(ep_rewards[-STATS_EVERY:])/len(ep_rewards[-STATS_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
		aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
		print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()