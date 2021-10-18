import logging

import torch


class ScheduledOptimizer:
    """ wrapper for optimizer
    """

    def __init__(self, optimizer, params, num_total_steps=None):
        self.optimizer = optimizer
        self.schedule_type = params.lr_schedule_type
        self._step = 0
        self._epoch = 0
        self.base_lr = params.learning_rate
        self.num_total_steps = num_total_steps

        # either `num_warmup_steps` or `warmup_proportion` must be specified
        assert hasattr(params, "num_warmup_steps") ^ hasattr(
            params, "warmup_proportion"
        )

        if hasattr(params, "warmup_proportion"):
            self.num_warmup_steps = int(num_total_steps * params.warmup_proportion)
            logging.info(f"warmup #steps: {self.num_warmup_steps:d}")
        else:
            self.num_warmup_steps = params.num_warmup_steps

        self._lr = 0 if self.num_warmup_steps > 0 else self.base_lr

        logging.info(f"lr scheduling type: {self.schedule_type}")
        if self.schedule_type == "epdecay":
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

        if self.schedule_type == "epdecay":
            if self._step <= self.num_warmup_steps:
                new_lr = (self.base_lr / max(1.0, self.num_warmup_steps)) * self._step
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

        elif self.schedule_type == "lindecay":
            # `transformers.get_linear_schedule_with_warmup`
            if self._step <= self.num_warmup_steps:
                new_lr = (self.base_lr / max(1.0, self.num_warmup_steps)) * self._step
            else:
                new_lr = self.base_lr * max(
                    0.0,
                    float(self.num_total_steps - self._step)
                    / float(max(1.0, self.num_total_steps - self.num_warmup_steps)),
                )

        if new_lr != self._lr:
            # set optimizer's learning rate
            for param in self.optimizer.param_groups:
                param["lr"] = new_lr

        self._lr = new_lr
        self.optimizer.step()

    def update_epoch(self):
        self._epoch += 1

        if self.schedule_type == "epdecay":
            if self._epoch >= self.lr_decay_start_epoch:
                new_lr = self._lr * self.lr_decay_rate
                # set optimizer's learning rate
                for param in self.optimizer.param_groups:
                    param["lr"] = new_lr
                logging.info(f"learning rate decreased: {self._lr:.6f} -> {new_lr:.6f}")
                self._lr = new_lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "_step": self._step,
            "_epoch": self._epoch,
            "base_lr": self.base_lr,
            "_lr": self._lr,
            "num_warmup_steps": self.num_warmup_steps,
            "num_total_steps": self.num_total_steps,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            elif key == "num_total_steps" and value is not None:
                assert self.num_total_steps == value
            else:
                setattr(self, key, value)


def optimizer_to(optimizer, device):
    for state in optimizer.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return optimizer


def get_optimizer_params_nodecay(model_named_params: list, weight_decay: float):
    nodecay_keys = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [
                p
                for n, p in model_named_params
                if not any(nd in n for nd in nodecay_keys)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model_named_params if any(nd in n for nd in nodecay_keys)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_params
