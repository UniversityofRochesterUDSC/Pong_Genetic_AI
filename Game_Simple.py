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


WIDTH = 1000
#HEIGHT = 400
HEIGHT = 600
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
score_ball_y=0
score_paddle1_y= 0
generation= 1
last_mean_score = 0

window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Daylight Pong')
#screen.set_alpha(None)
pygame.event.set_allowed([QUIT])
    

def ball_init(right):
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    rand_num = np.random.uniform(np.pi/2 +.4,np.pi*3/2-0.4)
    vert= int(10*np.sin(rand_num))
    horz= int(10*np.cos(rand_num))

    
    #horz = random.randrange(1,3)*-1
    #vert = random.randrange(1,3)*(1-2*int(np.random.rand()>.5))

    #if right == False:
    #    horz = - horz

    ball_vel = [horz, vert]
    #print(ball_vel)


def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel, l_score, r_score  # these are floats
    global score1, score2  # these are ints
    paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
    
    l_score = 0
    r_score = 0
    ball_init(True)
   # if random.randrange(0, 2) == 0:
    #    ball_init(True)
    #else:
    #    ball_init(False)


def draw(canvas,do_render):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score, paddle1_vel, score_ball_y,score_paddle1_y,generation, last_mean_score
   
    if(do_render):
        canvas.fill(BLACK)
        pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1)
        pygame.draw.circle(canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)

    if abs(paddle1_vel) > 10:
        paddle1_vel = paddle1_vel/abs(paddle1_vel)*10

    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel

    
    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    if(do_render):
        pygame.draw.circle(canvas, ORANGE, ball_pos, 20, 0)
        pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
                                            [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                            [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                            [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
  
    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]

    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(int(paddle1_pos[1]) - HALF_PAD_HEIGHT,
                                                                                 int(paddle1_pos[1]) + HALF_PAD_HEIGHT, 1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
        
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1

        score_ball_y = ball_pos[1]
        score_paddle1_y = paddle1_pos[1]
        
        print('R scored')
        ball_init(True)


    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        
        print('L scored Good Job')
        ball_init(False)

    if(do_render):
        myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
        label1 = myfont1.render("Score " + str(l_score), 1, (255, 255, 0))
        canvas.blit(label1, (50, 20))
        
        myfont2 = pygame.font.SysFont("Comic Sans MS", 10)
        label2 = myfont2.render("Generation " + str(generation), 1, (255, 255, 0))
        canvas.blit(label2, (200, 20))
        
        myfont2 = pygame.font.SysFont("Comic Sans MS", 10)
        label3 = myfont2.render("Last Performance Score " + str(last_mean_score), 1, (255, 255, 0))
        canvas.blit(label3, (400, 20))
    #print(paddle1_pos[1])


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


    
class AI:
    n_inputs = 2
    n_hidden_1 = 5
    n_hidden_2 = 5
    n_outputs = 3
    
    mu = 0.01
    
    inp_to_hid_weights = np.zeros(n_inputs*n_hidden_1).reshape(n_hidden_1, n_inputs)
    hid_to_hid_weights = np.zeros(n_hidden_1*n_hidden_2).reshape(n_hidden_2, n_hidden_1)
    hid_to_out_weights = np.zeros(n_outputs*n_hidden_2).reshape(n_outputs, n_hidden_2)
    
    def __init__(self):
        self.inp_to_hid_weights = 1-2*np.random.rand(self.n_inputs*self.n_hidden_1).reshape(self.n_hidden_1, self.n_inputs)
        self.hid_to_hid_weights = 1-2*np.random.rand(self.n_hidden_1*self.n_hidden_2).reshape(self.n_hidden_2, self.n_hidden_1)
        self.hid_to_out_weights = 1-2*np.random.rand(self.n_hidden_2*self.n_outputs).reshape(self.n_outputs, self.n_hidden_2)
    
    @classmethod
    def from_weights(cls,w1,w2,w3):
            
            ai = cls()
            ai.inp_to_hid_weights = w1
            ai.hid_to_hid_weights = w2
            ai.hid_to_out_weights = w3
            return ai
            
    def spawn(self,n_children):
        children = []
        for i in range(n_children):
            #mu = self.mutation_const + self.mutation_const*np.random.rand()*0.1
            mu= 1/10
            
            w1= self.randomize_fraction(self.inp_to_hid_weights,mu)
            w2= self.randomize_fraction(self.hid_to_hid_weights,mu)
            w3= self.randomize_fraction(self.hid_to_out_weights,mu)
            ai = AI.from_weights(w1,w2,w3)
            children.append(ai)
            
        return children
        
    def randomize_fraction(self,w,fract):
            num_w = w.shape[0]* w.shape[1]
            #print(num_w)
            mu = fract
            rand_w_vec = np.array([0]*(num_w-round(num_w*mu)) + [1]*(round(num_w*mu))).reshape(len(w[:,0]), len(w[0,:]))
            #print(w)
            np.random.shuffle(rand_w_vec)
            rand_part = np.random.rand(num_w).reshape(len(w[:,0]), len(w[0,:]))*rand_w_vec
            w = w*abs(rand_w_vec-1)- rand_part
            return w
    #w=np.array([list(range(100))]).reshape(10,10)
    #w2 = (np.array(list(range(100)))*-1).reshape(10,10)
    
    #ai1 = AI.from_weights(w,w,w)
    #ai2 = AI.from_weights(w2,w2,w2)
    
    #ai3= ai1.sex(ai2,1)
    def sex(self,ai_B,n_children):
        children = []
        for i in range(n_children):
            mu = 1/50
            w1_A= self.inp_to_hid_weights
            w2_A= self.hid_to_hid_weights
            w3_A= self.hid_to_out_weights
            w1_B= ai_B.inp_to_hid_weights
            w2_B= ai_B.hid_to_hid_weights
            w3_B= ai_B.hid_to_out_weights
            
            num_w1 = self.inp_to_hid_weights.shape[0]*self.inp_to_hid_weights.shape[1]
            num_w2 = self.hid_to_hid_weights.shape[0]*self.hid_to_hid_weights.shape[1]
            num_w3 = self.hid_to_out_weights.shape[0]*self.hid_to_out_weights.shape[1]
            
            rand_w1_vec = np.array([0]*(num_w1-round(num_w1*1/2)) + [1]*(round(num_w1*1/2))).reshape(len(w1_A[:,0]), len(w1_A[0,:]))
            np.random.shuffle(rand_w1_vec)
            rand_w2_vec = np.array([0]*(num_w2-round(num_w2*1/2)) + [1]*(round(num_w2*1/2))).reshape(len(w2_A[:,0]), len(w2_A[0,:]))
            np.random.shuffle(rand_w2_vec)
            rand_w3_vec = np.array([0]*(num_w3-round(num_w3*1/2)) + [1]*(round(num_w3*1/2))).reshape(len(w3_A[:,0]), len(w3_A[0,:]))
            np.random.shuffle(rand_w3_vec)
            
            
            
            w1 = w1_A*abs(rand_w1_vec-1) + rand_w1_vec*w1_B
            w2 = w2_A*abs(rand_w2_vec-1) + rand_w2_vec*w2_B
            w3 = w3_A*abs(rand_w3_vec-1) + rand_w3_vec*w3_B
                
            w1= self.randomize_fraction(w1,mu)
            w2= self.randomize_fraction(w2,mu)
            w3= self.randomize_fraction(w3,mu)
            #print(w1,w2,w3)
            ai = AI.from_weights(w1,w2,w3)
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
        hidden_1_out = np.array([self.activation_function(summed) for summed in summed_weighted_inputs])
        summed_weighted_hidden_1 =  np.matmul(self.hid_to_hid_weights,hidden_1_out)
        hidden_2_out = np.array([self.activation_function(summed) for summed in summed_weighted_hidden_1])
        summed_weighted_hidden_2 =  np.matmul(self.hid_to_out_weights,hidden_2_out)
        output = np.array([self.activation_function(summed) for summed in summed_weighted_hidden_2])
        return output
        
ai1 = AI()
ai2 = AI()




brawl_count= 0
rend_state = False

ai1.inp_to_hid_weights
ai1.respond([123,100],23)

def fitness_score(AI,rend_state):
    init()
    count =0 
    while True:
        global paddle1_vel, paddle2_vel, l_score, score_ball_y, score_paddle_y
        
        
      
        draw(window,rend_state)
        if(l_score+r_score is not l_score):
            #print(score_ball_y)
            l_score = l_score* (1-((score_ball_y - score_paddle1_y)**2)/HEIGHT**2)
            print(l_score)
            return [l_score,r_score]
        

        if(count %1 ==0):
            resp1 = AI.respond(np.array(ball_pos),np.array(paddle1_pos[1]))
            resp1 = np.array(resp1)/max(resp1) == 1
            
            if(resp1[0]):
                paddle1_vel = -8
            if(resp1[1]):
                paddle1_vel= 0
            if(resp1[2]):
                paddle1_vel = 8
                #print('huh')
            
            
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                keydown(event)
            elif event.type == KEYUP:
                keyup(event)
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.display.update()
        fps.tick(1000)
        count+=1

fits_for_scores = []
def fit_population(ais, show):
    global last_mean_score
    fit_scores= []
    new_ais = []
    for i in range(len(ais)):
        #print(i)
        fit_scores.append(fitness_score(ais[i], show)[0])
        
    total = sum(np.array(fit_scores))
    probs = fit_scores/total
    print(max(fit_scores))
    fits_for_scores.append((np.mean(fit_scores)))
    last_mean_score  = np.mean(np.array(fits_for_scores))
    print(probs)
    choices= np.random.choice(list(range(len(ais))),size=(101,),p=probs , replace=True)
    
    for i in range(len(choices)-1):
        new_ais.extend(ais[choices[i]].sex(ais[choices[i+1]],1))
    
    return new_ais


def champion_finder(gens):
    global generation
    ais=[]
    for i in range(120):
        ais.append(AI())
        
    for i in range(gens):
        new_ais = fit_population(ais, i>-1)
        ais = new_ais
        generation+=1
    return ais

ais= champion_finder(50)

#fitness_score(ais[2],True)
fit_population(ais,True)