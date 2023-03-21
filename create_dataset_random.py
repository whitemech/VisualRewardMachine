import random
import os
import cv2
from pygame.locals import *
from ImageEnvironment import GridWorldEnv
from MooreMachine import MinecraftMoore

MAX_LENGTH = 30
MIN_LENGTH = 15
EXECUTIONS = 20

dataset_path = "dataset_whole"
minecraft_machine = MinecraftMoore

Env = GridWorldEnv(minecraft_machine, "human", train=False)

i = 0
while i<EXECUTIONS:
    obs, reward, info = Env.reset()    
    print("Execution "+str(i))
    path=os.path.join(dataset_path, "episode_"+str(i)+"/")
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    terminated = False
    obs_list = []
    reward_list = []
    action_record = []
    last_reward = 0
    for k in range(MAX_LENGTH):
        print("Action "+str(k))
        action = random.randint(0,3)
        action_record.append(action)
        obs, last_reward, info, terminated = Env.step(action)
        obs_list.append(obs)
        reward_list.append(last_reward)
 
        path_ = path
        for o in range(len(obs_list)):
            cv2.imwrite(path_+"img"+str(o)+"_"+str(reward_list[o])+".jpg", obs_list[o])
    i+=1
