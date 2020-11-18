import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import numpy as np

class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=45, max_acceleration=0.05):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 0.3
        self.max_braking = max_acceleration
        self.free_deceleration = 0.001
        self.acceleration = 0.0
        self.steering = 0.0
        self.braking = 0.0

    def update(self):
        self.velocity += (self.acceleration - self.braking - self.free_deceleration , 0)
        self.velocity.x = max(0, min(self.velocity.x, self.max_velocity))
        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle)
        self.angle += degrees(angular_velocity)


class Game:
    def __init__(self):
        pygame.init()
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.ticks = 60
        self.exit = False
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        self.car_image = pygame.image.load(image_path)
        self.numberOfCars = 100
        self.cars = []
        for i in range(self.numberOfCars):
            self.cars.append(Car(10,30))
        self.ppu = 16
        self.draw()

    def step(self, action, render = True):
       
        
        # while not self.exit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        for i in range(self.numberOfCars):
            self.cars[i].steering     = action[i][0]*self.cars[i].max_steering
            self.cars[i].acceleration = action[i][1]*self.cars[i].max_acceleration
            self.cars[i].braking      = action[i][2]*self.cars[i].max_braking
            self.cars[i].update()
        
        if render:
            self.draw()
        
        # Drawing
        
    def draw(self):
        self.screen.fill((0, 0, 0))

        for i in range(self.numberOfCars):
            rotated = pygame.transform.rotate(self.car_image, self.cars[i].angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, self.cars[i].position * self.ppu - (rect.width / 2, rect.height / 2))
        pygame.display.flip()

            # self.clock.tick(self.ticks)
        


if __name__ == '__main__':
    game = Game()
    action = [0.0, 0.0, 0.0]
    
    i = 0
    while i < 2500:
        i += 1 
        action = np.random.rand( game.numberOfCars, 3)
        # action[:, 0] *=
        action[:, 0] -=0.5
        # action = [1.0, 1.0, 0.5]
        game.step(action)
    pygame.quit()