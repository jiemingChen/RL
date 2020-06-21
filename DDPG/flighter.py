from rlschool import make_env 

from agent import Agent
from algorithm import DDPG
from model import Model 
import torch
from pdb import set_trace

ACTOR_LR = 0.0002  
CRITIC_LR = 0.001   

GAMMA = 0.99   
TAU = 0.001       

MEMORY_WARMUP_SIZE = 1e4     
REWARD_SCALE = 0.01
BATCH_SIZE = 256         
TRAIN_TOTAL_STEPS = 1e6   
TEST_EVERY_STEPS = 1e4 

BUFFER_SIZE = int(1e6)  # replay buffer size

def action_mapping(action, low_val, high_val):
    action =  action * ((high_val-low_val)/2.0) + low_val 
    return action


def run_episode(env, agent):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        action = agent.act(obs)
        action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])       
        
        next_obs, reward, done, _ = env.step(action)
        agent.step(obs, action, REWARD_SCALE*reward, next_obs, done)

        obs = next_obs
        total_reward += reward
        steps += 1

        if done:
            break

    return total_reward, steps           

env = make_env("Quadrotor", task="hovering_control")
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
set_trace()
model = Model(state_size=obs_dim, action_size=act_dim)
target_model = Model(state_size=obs_dim, action_size=act_dim)

alg =  DDPG(model,target_model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = Agent(alg, BUFFER_SIZE, BATCH_SIZE, seed=10)


total_steps = 0
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent)
    total_steps += steps
    print('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

    if total_steps % TEST_EVERY_STEPS ==0: # 每隔一定step数，评估一次模型
        torch.save(agent.alg.model.actor_model.state_dict(), f'flighter_actor_{total_steps}.pth')
        torch.save(agent.alg.model.critic_model.state_dict(), f'flighter_actor_{total_steps}.pth')