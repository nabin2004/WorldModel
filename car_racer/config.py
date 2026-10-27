"""
Configuration and hyperparameters for Mini World Models (Car Racing).
"""
import os
import torch

class Config:
    # Random seed
    SEED = 0

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment parameters
    ENV_SIZE = 24
    ENV_ACTIONS = 3  # 0: left, 1: straight, 2: right
    ENV_R = 8.0
    ENV_HALF_W = 2.2
    ENV_SPEED = 1.5
    EPISODE_STEPS = 150

    # Dataset collection
    DATASET_EPISODES = 150
    DATASET_HORIZON = 150

    # VAE (Vision Model V)
    LATENT_DIM = 16  # Z
    VAE_LR = 2e-3
    VAE_BATCH_SIZE = 256
    VAE_EPOCHS = 30
    VAE_CAR_WEIGHT = 30.0
    VAE_LATENT_NOISE = 0.10

    # MDN-RNN (Memory Model M)
    RNN_HIDDEN_DIM = 160  # H
    NUM_GAUSSIANS = 5     # K
    MDN_LR = 1e-3
    MDN_BATCH_SIZE = 16
    MDN_EPOCHS = 30
    MDN_REWARD_WEIGHT = 10.0
    MDN_TEMPERATURE = 0.7

    # Controller (C)
    CONTROLLER_PARAMS = 51  # 16 * 3 + 3
    ES_NUM_PAIRS = 12       # 24 candidates total per gen
    ES_GENERATIONS = 20
    ES_SIGMA = 0.5
    ES_LR = 0.25
    ES_SEEDS_PER_GEN = 6
    ES_DREAM_HORIZON = 40
    ES_EVAL_HORIZON = 100

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
