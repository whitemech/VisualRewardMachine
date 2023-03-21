import pygame
from pygame.locals import *
from ImageEnvironment import GridWorldEnv
from MooreMachine import MinecraftMoore

minecraft_machine = MinecraftMoore

if __name__=="__main__":
    
    Env = GridWorldEnv(minecraft_machine, "human", False)
    obs, reward, info, termination = Env.reset()
    print(obs)

    # TESTING MOVEMENT
    while True: #and not termination:
        for e in pygame.event.get():
            if e.type == QUIT:
                Env.close()
            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    Env.close()
                elif e.key == K_s:
                    obs, reward, info, termination = Env.step(0)
                    print(obs)
                    print("termination: "+str(termination))
                elif e.key == K_d:
                    obs, reward, info, termination = Env.step(1)
                    print(obs)
                    print("termination: "+str(termination))
                elif e.key == K_w:
                    obs, reward, info, termination = Env.step(2)
                    print(obs)
                    print("termination: "+str(termination))
                elif e.key == K_a:
                    obs, reward, info, termination = Env.step(3)
                    print(obs)
                    print("termination: "+str(termination))
