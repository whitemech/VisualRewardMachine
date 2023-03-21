import pygame
import os
import cv2
from pygame.locals import *
from ImageEnvironment import GridWorldEnv
from MooreMachine import MinecraftMoore

EXECUTIONS = 20
MAX_LENGTH = 30

dataset_path = "dataset_whole"
minecraft_machine = MinecraftMoore

Env = GridWorldEnv(minecraft_machine, "human", train=False)

for i in range(EXECUTIONS):
    obs, reward, info = Env.reset()
    print("Execution "+str(i))
    path=os.path.join(dataset_path, "episode_"+str(EXECUTIONS+i)+"/")
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    terminated = False
    obs_list = []
    reward_list = []
    action_record = []
    info_record = []
    j=0
    while(True and j<MAX_LENGTH):
        for e in pygame.event.get():
            if e.type == QUIT:
                Env.close()
            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    Env.close()
                elif e.key == K_s:
                    obs, reward, info, terminated = Env.step(0)
                    obs_list.append(obs)
                    reward_list.append(reward)
                    info_record.append(info)
                    j+=1
                elif e.key == K_d:
                    obs, reward, info, terminated = Env.step(1)
                    obs_list.append(obs)
                    reward_list.append(reward)
                    info_record.append(info)
                    j+=1
                elif e.key == K_w:
                    obs, reward, info, terminated = Env.step(2)
                    obs_list.append(obs)
                    reward_list.append(reward)
                    info_record.append(info)
                    j+=1
                elif e.key == K_a:
                    obs, reward, info, terminated = Env.step(3)
                    obs_list.append(obs)
                    reward_list.append(reward)
                    info_record.append(info)
                    j+=1

    path_ = path
    try:
        os.mkdir(path_)
    except OSError as error:
        print(error)
    for o in range(len(obs_list)):
        cv2.imwrite(path_+"img"+str(o)+"_"+str(reward_list[o])+".jpg", obs_list[o])