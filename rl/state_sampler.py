import numpy as np
from utils.utils import compute_epoch

class StateSampler:

    def __init__(self, discrete_state_space, state_fn, state_fn_args, preprocess_fn=lambda x: x, batch_size=64):
        self.S = discrete_state_space
        self.N = len(self.S)
        self.batch_size = batch_size
        self.batch_idx = 0

        self.state_fn = state_fn
        self.state_fn_args = state_fn_args
        self.preprocess_fn = preprocess_fn
        self.epoch_cnt = 0

    def reset_stats(self):
        self.batch_idx = 0
        self.epoch_cnt = 0

    def next_batch(self):
        self.batch_pre_process()
        x = [getattr(self.S[idx], self.state_fn)(**self.state_fn_args) for idx in np.random.choice(len(self.S), self.batch_size)]
        self.batch_idx += 1
        self.batch_post_process()
        return self.preprocess_fn(x)

    def curr_epoch(self):
        return self.epoch_cnt

    def batch_pre_process(self):
        pass

    def batch_post_process(self):
        epoch_cnt = compute_epoch(self.batch_idx, self.batch_size, self.N)
        if epoch_cnt != self.epoch_cnt:
            self.epoch_cnt = epoch_cnt
            self.epoch_changed = True
        else:
            self.epoch_changed = False

    def batch_count(self):
        return self.batch_idx

    def epoch_done(self):
        return self.epoch_changed or (self.curr_epoch() == 0 and self.batch_idx == 1)

    def get_batch_size(self):
        return self.batch_size