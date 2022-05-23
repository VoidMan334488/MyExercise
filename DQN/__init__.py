from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random

GAME = 'flappy bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000.
EXPLORE = 3.0e6
FINAL_EPSILON = 1.0e-4
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1