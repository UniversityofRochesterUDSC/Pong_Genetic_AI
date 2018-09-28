# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:55:57 2018

@author: ethan
"""

# Developer : Hamdy Abou El Anein

import random
import numpy as np
import pygame
import sys
from pygame import *


pygame.init()
fps = pygame.time.Clock()


WHITE = (255, 255, 255)
ORANGE = (255,140,0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
ball_pos = [0, 0]
ball_vel = [0, 0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0


window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Daylight Pong')
#screen.set_alpha(None)
pygame.event.set_allowed([QUIT])
    

def ball_init(right):
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    #horz = random.randrange(2,5)*-1
    #vert = random.randrange(1,2)*(1-2*int(np.random.rand()>.5))
    vert = 3*np.sin((np.random.uniform(-.7,.7)*np.pi/2))
    horz= -3*np.cos((np.random.uniform(.2,.8)*np.pi/2))
    #if right == False:
    #    horz = - horz

    ball_vel = [horz, -vert]


def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel, l_score, r_score  # these are floats
    global score1, score2  # these are ints
    paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
    paddle2_pos = [WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT //2]
    l_score = 0
    r_score = 0
    ball_init(True)
   # if random.randrange(0, 2) == 0:
    #    ball_init(True)
    #else:
    #    ball_init(False)


def draw(canvas,do_render):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score
        
    if(do_render):
        canvas.fill(BLACK)
        pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1)
        pygame.draw.circle(canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)


    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel

    if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
        paddle2_pos[1] += paddle2_vel


    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    if(do_render):
        pygame.draw.circle(canvas, ORANGE, ball_pos, 20, 0)
        pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
                                            [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                            [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                            [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT],
                                            [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                            [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                            [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)


    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]


    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(paddle1_pos[1] - HALF_PAD_HEIGHT,
                                                                                 paddle1_pos[1] + HALF_PAD_HEIGHT, 1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)

    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(
            paddle2_pos[1] - HALF_PAD_HEIGHT, paddle2_pos[1] + HALF_PAD_HEIGHT, 1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        ball_init(False)

    if(do_render):
        myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
        label1 = myfont1.render("Score " + str(l_score), 1, (255, 255, 0))
        canvas.blit(label1, (50, 20))
    
        myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
        label2 = myfont2.render("Score " + str(r_score), 1, (255, 255, 0))
        canvas.blit(label2, (470, 20))


def keydown(event):
    global paddle1_vel, paddle2_vel

    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8
    elif event.key == K_z:
        paddle1_vel = -8
    elif event.key == K_s:
        paddle1_vel = 8


def keyup(event):
    global paddle1_vel, paddle2_vel

    if event.key in (K_z, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0

"""
init()


while True:

    draw(window)
    
    
    if(ball_pos[1] > 200):
        paddle1_vel  =8 
    else:
        paddle1_vel = -8
        
    for event in pygame.event.get():

        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fps.tick(60*10)
"""
def same():
    
    init()
    
    
    while True:
    
        draw(window)
        
        
        if(ball_pos[1] > 200):
            paddle1_vel  =8 
        else:
            paddle1_vel = -8
            
        for event in pygame.event.get():
    
            if event.type == KEYDOWN:
                keydown(event)
            elif event.type == KEYUP:
                keyup(event)
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()
    
        pygame.display.update()
        fps.tick(60*10)




    
class AI:
    n_inputs = 2
    n_hidden = 3
    n_outputs = 1
    
    mutation_const = .8
    
    inp_to_hid_weights = np.zeros(n_inputs*n_hidden).reshape(n_hidden, n_inputs)
    hid_to_out_weights = np.zeros(n_inputs*n_hidden).reshape(n_hidden, n_inputs)
    
    def __init__(self):
        self.inp_to_hid_weights = 1-2*np.random.rand(self.n_inputs*self.n_hidden).reshape(self.n_hidden, self.n_inputs)
        self.hid_to_out_weights = 1-2*np.random.rand(self.n_hidden*self.n_outputs).reshape(self.n_outputs, self.n_hidden)
        self.mutation_const = 1
    
    @classmethod
    def from_weights(cls,w1,w2,mu):
            
            ai = cls()
            ai.inp_to_hid_weights = w1
            ai.hid_to_out_weights = w2
            ai.mutation_const = mu
            return ai
            
    def spawn(self,n_children):
        children = []
        for i in range(n_children):
            #mu = self.mutation_const + self.mutation_const*np.random.rand()*0.1
            mu=self.mutation_const

            
            w1= self.randomize_fraction(self.inp_to_hid_weights,mu)
            w2= self.randomize_fraction(self.hid_to_out_weights,mu)
            ai = AI.from_weights(w1,w2,mu)
            children.append(ai)
            
        return children
        
    def randomize_fraction(self,w,fract):
            num_w = w.shape[0]* w.shape[1]
            print(num_w)
            mu = fract
            rand_w_vec = np.array([0]*(num_w-round(num_w*mu)) + [1]*(round(num_w*mu))).reshape(len(w[:,0]), len(w[0,:]))
            print(w)
            np.random.shuffle(rand_w_vec)
            rand_part = np.random.rand(num_w).reshape(len(w[:,0]), len(w[0,:]))*rand_w_vec
            w = w*abs(rand_w_vec-1)- rand_part
            return w
        
    def sex(self,ai_B,n_children):
        children = []
        for i in range(n_children):
            mu = 1/5
            w1_A= self.inp_to_hid_weights
            w2_A= self.hid_to_out_weights
            w1_B= ai_B.inp_to_hid_weights
            w2_B= ai_B.hid_to_out_weights
            
            num_w1 = self.inp_to_hid_weights.shape[0]*self.inp_to_hid_weights.shape[1]
            num_w2 = self.hid_to_out_weights.shape[0]*self.hid_to_out_weights.shape[1]
            
            rand_w1_vec = np.array([0]*(num_w1-round(num_w1*1/2)) + [1]*(round(num_w1*1/2))).reshape(len(w1_A[:,0]), len(w1_A[0,:]))
            np.random.shuffle(rand_w1_vec)
            rand_w2_vec = np.array([0]*(num_w2-round(num_w2*1/2)) + [1]*(round(num_w2*1/2))).reshape(len(w2_A[:,0]), len(w2_A[0,:]))
            np.random.shuffle(rand_w2_vec)
            
            print(w2_A)
            print(rand_w2_vec)
            print(w2_B)
            w1 = w1_A*abs(rand_w1_vec-1) + rand_w1_vec*w1_B
            w2 = w2_A*abs(rand_w2_vec-1) + rand_w2_vec*w2_B
                
            w1= self.randomize_fraction(w1,mu)
            w2= self.randomize_fraction(w2,mu)
            
            ai = AI.from_weights(w1,w2,mu)
            children.append(ai)
        return children
        
        
    def activation_function(self,x):
        return x
    
    def respond(self,ball_pos,pos):
        ball_px, ball_py = ball_pos
        #ball_vx, ball_vy = ball_vel
        #print([ball_px,ball_py,ball_vx, ball_vy,pos])
        
        inputs = np.array([ball_py,pos])
        
        summed_weighted_inputs = np.matmul(self.inp_to_hid_weights,inputs)
        
        hidden_out = np.array([self.activation_function(summed) for summed in summed_weighted_inputs])
        
        summed_weighted_hidden =  np.matmul(self.hid_to_out_weights,hidden_out)
        
        output = np.array([self.activation_function(summed) for summed in summed_weighted_hidden])
        return output
        
ai1 = AI()
ai2 = AI()




brawl_count= 0
rend_state = False

ai1.inp_to_hid_weights
ai1.respond([123,100],23)

def brawl(l_AI, r_AI):
    global brawl_count, rend_state
    
    if(brawl_count>40):
        rend_state = True
    print(brawl_count)
    brawl_count +=1
    
    init()
    count =0 
    while True:
        global paddle1_vel, paddle2_vel
        
        draw(window,rend_state)
        
        if(count > 5):
            return [l_score,r_score]

        if(count %10 ==0):
            resp1 = l_AI.respond(np.array(ball_pos),np.array(paddle1_pos[1]))[0] > .5
            
            #Flipping the x velocity and the x postion to mirror the cordinates
            ball_pos_r= np.array(ball_pos)
            ball_pos_r[0] = WIDTH-ball_pos_r[0]
            ball_vel_r= np.array(ball_vel)
            ball_vel_r[0] = -ball_vel_r[0]
            
            resp2 = r_AI.respond(ball_pos_r,paddle2_pos[1])[0] > .5
            
            #print(resp1)
            if(resp1):
                paddle1_vel = 8
            else:
                paddle1_vel = -8
           
                
            if(resp2):
                paddle2_vel = 8
            else:
                paddle2_vel = -8
             
            
            
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                keydown(event)
            elif event.type == KEYUP:
                keyup(event)
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.display.update()
        fps.tick(60*10)
        count+=1
    
def pitch(ai,n):
    global brawl_count, rend_state
    
    if(brawl_count>100):
        rend_state = True
    print(brawl_count)
    brawl_count +=1
    
    init()
    count =0 
    while True:
        global paddle1_vel, paddle2_vel
        
        draw(window,rend_state)
        
        if(l_score + r_score == n):
            score = l_score/(n)
            return score
        
        if(count %10 ==0):
            resp1 = ai.respond(np.array(ball_pos),np.array(ball_vel),np.array(paddle1_pos[1]))
            
        
            #print(resp1)
            if(resp1[0]):
                paddle1_vel = 8
            elif(resp1[1]):
                paddle1_vel = -8
            elif(resp1[2]):
                paddle1_vel = 0

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    keydown(event)
                elif event.type == KEYUP:
                    keyup(event)
                elif event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                    
        pygame.display.update()
        fps.tick(60*10*10)
        count+=1
    
    
def fittest(ais,use_pitch):
    ai1 = ais[0]
    scores = []
    scores.append(1)
    w_index = 0
    best_score = 0
    for i in range(1,len(ais)):
        if(use_pitch):
          score = pitch(ais[i],10)
          if(score > best_score):
              w_index = i
              best_score = score
              ai1 = ais[i]
        else:
            ai2= ais[i]
            round_scores = brawl(ai2, ai1)
            scores.append(round_scores[0])
    
    return scores


def champion_finder(gens):
    ai= AI()
    ais = ai.spawn(10)
    for i in range(gens):
        
        scores = fittest(ais,False)
        
        indexes = np.sort(scores.top(2))
        ais[indexes[0]].sex(ais[indexes[1]])
        
   
    return ai

ai_w,ai_index = champion_finder(200)


#brawl(ai1,ai2)

"""
    if(resp1[0]):
        paddle1_vel = 8
    elif(resp1[1]):
        paddle1_vel = -8
    elif(resp1[2]):
        paddle1_vel = 0
        
    if(resp2[0]):
        paddle2_vel = 8
    elif(resp2[1]):
        paddle2_vel = -8
    elif(resp2[2]):
        paddle2_vel = 0
"""