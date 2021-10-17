import logging

import torch


class ScheduledOptimizer:
    """ wrapper for optimizer
    """

    def __init__(self, optimizer, params):
        self.optimizer = optimizer
        self.schedule_type = params.lr_schedule_type
        self._step = 0
        self._epoch = 0
        self.base_lr = params.learning_rate
        self.num_warmup_steps = params.num_warmup_steps

        self._lr = 0 if self.num_warmup_steps > 0 else self.base_lr

        logging.info(f"lr scheduling type: {self.schedule_type}")
        if self.schedule_type == "linear":
            self.lr_decay_start_epoch = params.lr_decay_start_epoch
            self.lr_decay_rate = params.lr_decay_rate
        elif self.schedule_type == "noam":
            self.model_dim = params.enc_hidden_size

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1

        new_lr = None

        if self.schedule_type == "linear":
            if self.num_warmup_steps > 0 and self._step <= self.num_warmup_steps:
                new_lr = (self.base_lr / self.num_warmup_steps) * self._step
            else:
                new_lr = self._lr  # keep self._lr after num_warmup_steps

        elif self.schedule_type == "noam":
            new_lr = (
                self.base_lr
                * self.model_dim ** (-0.5)
                * min(
                    self._step ** (-0.5), self._step * self.num_warmup_steps ** (-1.5)
                )
            )

        if new_lr != self._lr:
            # set optimizer's learning rate
            for param in self.optimizer.param_groups:
                param["lr"] = new_lr

        self._lr = new_lr
        self.optimizer.step()

    def update_epoch(self):
        self._epoch += 1

        if self.schedule_type == "linear":
            if self._epoch >= self.lr_decay_start_epoch:
                new_lr = self._lr * self.lr_decay_rate
                # set optimizer's learning rate
                for param in self.optimizer.param_groups:
                    param["lr"] = new_lr
                logging.info(f"learning rate decreased: {self._lr:.6f} -> {new_lr:.6f}")
                self._lr = new_lr

        # do nothing if self.shedule_type == "noam"

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "_step": self._step,
            "_epoch": self._epoch,
            "base_lr": self.base_lr,
            "_lr": self._lr,
            "num_warmup_steps": self.num_warmup_steps,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def optimizer_to(optimizer, device):
    for state in optimizer.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return optimizer
