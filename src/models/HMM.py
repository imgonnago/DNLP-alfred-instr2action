import json
import yaml
import numpy as np
from src.make_vocab import load_vocab, load_data, encode
from src.utils import load_config

class HMM():
    def __init__(self, k : float = 0.1):
        cfg = load_config()
        self.words_itos, self.words_stoi = load_vocab("words")
        self.actions_itos, self.actions_stoi = load_vocab("actions")
        self.obs_size = len(self.words_itos)
        self.state_size = len(self.actions_itos)
        self.k = k

        special_size = len(cfg["tokens"]["action_special"])
        self.real_states = np.arange(special_size, self.state_size)
        excluded = [self.words_stoi["<pad>"], self.words_stoi["<sep>"]]
        mask = np.ones(self.obs_size, dtype=bool)
        mask[excluded] = False
        self.obs_cols = np.where(mask)[0]

    def count(self, data):
        self.pi_count = np.zeros(self.state_size)
        self.A_count = np.zeros(self.state_size, self.state_size)
        self.B_count = np.zeros(self.obs_size, self.state_size)

        for actions, steps in data:
            self.pi_count[actions[0]] += 1
            for t in range(1, len(self.state_size)):
                self.A_count[actions[t - 1], actions[t]] += 1
            for a, sent in zip(actions, steps):
                for w in sent:
                    self.B_count[a, w] += 1

    def smooth(self):
        k = self.k
        rs = self.real_states
        oc = self.obs_cols

        pi = np.zeros(self.state_size)
        pi[rs] = self.pi_count[rs] + k
        pi /= pi.sum()

        A = np.zeros((self.state_size, self.state_size))
        sub = self.A_count[np.ix_(rs,rs)] + k
        A[np.ix_(rs, rs)] = sub / sub.sum(axis=1, keepdims=True) 

        B = np.zeros((self.obs_size, self.state_size))
        sub = self.B_count[np.ix_(oc, rs)] + k
        B[np.ix_(oc,rs)] = sub / sub.sum(axis=1, keepdims=True)

        self.log_pi = np.log(pi)
        self.log_state = np.log(A)
        self.log_obs = np.log(B)

    def fit(self, data):
        self.count(data)
        self.smooth()

def main():
    model = HMM(k=0.1)
    train_data = load_data("train")
    data = [
        encode

    ]