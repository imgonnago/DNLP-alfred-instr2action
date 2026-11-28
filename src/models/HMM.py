import json
import yaml
import numpy as np
from src.make_vocab import load_vocab, load_data, encode, decode, tokenize
from src.utils import load_config

class HMM():
    def __init__(self, k : float = 0.1):
        cfg = load_config()
        self.words_itos, self.words_stoi = load_vocab("words")
        self.actions_itos, self.actions_stoi = load_vocab("actions")
        self.obs_size = len(self.words_itos)
        self.state_size = len(self.actions_itos)
        self.k = k

        special_size = len(cfg["tokens"]["actions_special"])
        self.real_states = np.arange(special_size, self.state_size)
        excluded = [self.words_stoi["<pad>"], self.words_stoi["<sep>"]]
        mask = np.ones(self.obs_size, dtype=bool)
        mask[excluded] = False
        self.obs_cols = np.where(mask)[0]

    def prepare(self, line):
        actions = encode(line["actions"], self.actions_stoi)
        steps = [encode(tokenize(s), self.words_stoi) for s in line["steps"]]
        return actions, steps

    def count(self, data):
        self.pi_count = np.zeros((self.state_size))
        self.A_count = np.zeros((self.state_size, self.state_size))
        self.B_count = np.zeros((self.state_size, self.obs_size))

        for actions, steps in data:
            self.pi_count[actions[0]] += 1
            for t in range(1, len(actions)):
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

        B = np.zeros((self.state_size, self.obs_size))
        sub = self.B_count[np.ix_(rs, oc)] + k
        B[np.ix_(rs,oc)] = sub / sub.sum(axis=1, keepdims=True)

        self.log_pi = np.log(pi)
        self.log_state = np.log(A)
        self.log_obs = np.log(B)

    def fit(self, data):
        self.count(data)
        self.smooth()

    def NLL(self, data):
        total, n_words = 0.0, 0
        for actions, steps in data:
            total += self.log_pi[actions[0]]
            for t in range(1, len(actions)):
                total += self.log_state[actions[t - 1], actions[t]]
            for a, sent in zip(actions, steps):
                total += self.log_obs[a, sent].sum()
                n_words += len(sent)
        return -total / n_words

    def viterbi(self, steps):
        T, S = len(steps), self.state_size

        emit = np.array([self.log_obs[:, sent].sum(axis=1) for sent in steps])

        delta = np.empty((T, S))
        back = np.zeros((T, S), dtype=int)

        delta[0] = self.log_pi + emit[0]
        for t in range(1, T):
            score = delta[t - 1][:, None] + self.log_state
            back[t] = score.argmax(axis=0)
            delta[t] = score.max(axis=0) + emit[t]

        path = [int(delta[-1].argmax())]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        path.reverse()
        return path

    def predict(self, sentence):
        steps = [encode(tokenize(s), self.words_stoi) for s in sentence]
        return decode(self.viterbi(steps), self.actions_itos)

    def save(self, path):
        np.savez(path, k = self.k, pi_count = self.pi_count,
                A_count = self.A_count, B_count = self.B_count)

    @classmethod
    def load(cls, path):
        ckpt = np.load(path)
        model = cls(k = float(ckpt["k"]))
        model.pi_count, model.A_count, model.B_count = ckpt["pi_count"], ckpt["A_count"], ckpt["B_count"]
        model.smooth()
        return models