from environment.car import Car
import pygame
from pygame.math import Vector2
import numpy as np
from config import Args, configure
import time

class Environment:
    def __init__(self, configs:Args):
        self.exit = False
        pygame.init()  
        self.configs = configs
        self.screen = pygame.display.set_mode((configs.cameraHeight, configs.cameraHeight))
        self.trackImage = pygame.image.load(configs.trackPath).convert_alpha()
        self.cars = pygame.sprite.Group()
        self.cameraPosition = Vector2(configs.startingPositionX - configs.cameraHeight//2,configs.startingPositionY - configs.cameraHeight//2)
        self.cameraPositions = []
        self.cameraPositions.append((self.cameraPosition.x, self.cameraPosition.y))
        
        

        for i in range(self.configs.numberOfCars):
            self.cars.add(Car(configs.startingPositionX , configs.startingPositionY - i//2, index = i, configs = self.configs, trackImage=  self.trackImage))
        
        self.draw()

    def step(self, action, render = True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        state = np.zeros((self.configs.numberOfCars, self.configs.numberOfLasers))
        rewards = np.zeros((self.configs.numberOfCars,))
        dead = np.zeros((self.configs.numberOfCars,))
        i = 0

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            action[:, 1] = 1.0
        if pressed[pygame.K_LEFT]:
            action[:, 0] = 1.0
        if pressed[pygame.K_RIGHT]:
            action[:, 0] = -1.0
        if pressed[pygame.K_SPACE]:
            action[:, 2] = 1.0

        
        bestCarDistance =  0
        for car in self.cars:
            state[i], dead[i], rewards[i], carDistance = car.update(action[i], self.screen)
            if carDistance > bestCarDistance:
                bestCarDistance = carDistance
                self.cameraPosition.x = car.position.x - self.configs.cameraHeight//2
                self.cameraPosition.y = car.position.y - self.configs.cameraHeight//2
            i += 1
        
        if len(self.cameraPositions) > 200:
            self.cameraPositions.pop(0)
        self.cameraPositions.append((self.cameraPosition.x, self.cameraPosition.y))
        mean = np.mean(self.cameraPositions, 0)
        self.cameraPosition =Vector2(int(mean[0]), int(mean[1]))

        if render:
            self.draw()
        
        return state, dead, rewards
        
    def reset(self):
        self.cars = pygame.sprite.Group()
        self.cameraPosition = Vector2(self.configs.startingPositionX - self.configs.cameraHeight//2,self.configs.startingPositionY - self.configs.cameraHeight//2)
        self.cameraPositions = []
        self.cameraPositions.append((self.cameraPosition.x, self.cameraPosition.y))
        for i in range(self.configs.numberOfCars):
            self.cars.add(Car(self.configs.startingPositionX , self.configs.startingPositionY, index = i, configs = self.configs, trackImage =  self.trackImage))
        
        self.draw()

    
    def draw(self):
        self.screen.fill(self.configs.bgColor)
        cropRect = (self.cameraPosition.x, self.cameraPosition.y, self.configs.cameraHeight, self.configs.cameraHeight)
        self.screen.blit(self.trackImage,(0,0),cropRect)
        for car in self.cars:
            car.draw(self.screen, self.cameraPosition)
        pygame.display.flip()
         
 


if __name__ == '__main__':
    
    
    config = configure()

    env = Environment(config)
    
    while True:
        action = np.random.randn(config.numberOfCars, 3)
        
        state, dead, rewards = env.step(action, render = config.render)

        if 0.0 not in dead:
            time.sleep(2)
            env.reset()

    