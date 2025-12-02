# config_consts.py - Stores all global constants for the TCS-26 Symplectic-Omni System

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, Flatten, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, GlobalMaxPooling1D, Concatenate, Multiply, BatchNormalization, Dot, Reshape, Dropout, Attention, Add, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import traceback
import random
import uuid
import json
import time

# --- TCS-26 Global Constants (Hyper-Advanced) ---
IMAGE_SHAPE = (128, 128, 3) 
DATA_INPUT_SIZE = 512
TS_STEPS = 10 
TS_DIM = 8 
SEQ_LEN = 50
SEQ_DIM = 64
GRAPH_DIM = 32
VOCAB_SIZE = 10000 
NUM_CLASSES = 10
FRONTAL_LOBE_UNITS = 2048
HYPER_LATENT_DIM = 4096 
NUM_PFC_CONTEXTS = 64 
RELATIONAL_EMB_DIM = 512 
CAUSAL_STATE_DIM = 256 
AXIOMATIC_DIM = 128 
MLC_OUTPUT_DIM = 5 
SYMPLECTIC_DIM = 256 # NEW: Phase space dimension (must be even for q, p)
LOSS_WEIGHT_SSTC = 0.95
TRAINING_BATCH_SIZE = 32 
