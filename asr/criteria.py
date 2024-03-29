import torch
from torch import nn


def to_onehot(label: torch.tensor, num_classes: int) -> torch.tensor:
    return torch.eye(num_classes)[label].to(label.device)


def to_onehot_lsm(
    labels: torch.tensor, num_classes: int, lsm_prob: float = 0.1
) -> torch.tensor:
    onehot = to_onehot(labels, num_classes)
    onehot_lsm = (1 - lsm_prob) * onehot + (lsm_prob / (num_classes - 1)) * (1 - onehot)

    return onehot_lsm


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self, vocab_size, lsm_prob=0, normalize_length=False, normalize_batch=True
    ):
        super(LabelSmoothingLoss, self).__init__()

        self.vocab_size = vocab_size
        self.lsm_prob = lsm_prob
        self.normalize_length = normalize_length
        self.normalize_batch = normalize_batch

    def forward(self, logits, ys, ylens):
        bs = logits.size(0)
        loss = 0
        onehot_lsm = to_onehot_lsm(ys, self.vocab_size, self.lsm_prob)

        for b in range(bs):
            ylen = ylens[b]
            loss_b = torch.sum(
                torch.log_softmax(logits[b, :ylen], dim=-1) * onehot_lsm[b, :ylen]
            )
            if self.normalize_length:
                loss_b /= ylen
            loss -= loss_b

        if self.normalize_batch:
            loss /= bs

        return loss


class DistillLoss(nn.Module):
    def __init__(
        self,
        vocab_size,
        soft_label_weight,
        lsm_prob=0,
        normalize_length=False,
        normalize_batch=True,
    ):
        super(DistillLoss, self).__init__()

        self.vocab_size = vocab_size
        self.soft_label_weight = soft_label_weight
        self.lsm_prob = lsm_prob
        self.normalize_length = normalize_length
        self.normalize_batch = normalize_batch

    def forward(self, logits, ys, soft_labels, ylens):
        bs = logits.size(0)
        hard_labels = to_onehot_lsm(ys, self.vocab_size, self.lsm_prob)
        loss = 0
        loss_soft = 0
        loss_hard = 0

        for b in range(bs):
            ylen = ylens[b]

            loss_soft_b = torch.sum(
                torch.log_softmax(logits[b, :ylen], dim=-1) * soft_labels[b, :ylen]
            )
            loss_hard_b = torch.sum(
                torch.log_softmax(logits[b, :ylen], dim=-1) * hard_labels[b, :ylen]
            )
            if self.normalize_length:
                loss_soft_b /= ylen
                loss_hard_b /= ylen

            # label interpolation
            loss_b = (
                self.soft_label_weight * loss_soft_b
                + (1 - self.soft_label_weight) * loss_hard_b
            )
            loss -= loss_b
            loss_soft -= loss_soft_b
            loss_hard -= loss_hard_b

        if self.normalize_batch:
            loss /= bs
            loss_soft /= bs
            loss_hard /= bs

        return loss, loss_soft, loss_hard


class CTCAlignDistillLoss(nn.Module):
    def __init__(
        self,
        vocab_size,
        blank_id=0,
        soft_label_weight=1.0,
        position="all",
        lsm_prob=0,
        normalize_length=True,
        normalize_batch=True,
    ):
        super(CTCAlignDistillLoss, self).__init__()

        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.soft_label_weight = soft_label_weight
        self.position = position
        self.lsm_prob = lsm_prob
        self.normalize_length = normalize_length
        self.normalize_batch = normalize_batch

    def forward(self, logits, ys, soft_labels, aligns, xlens, ylens):
        bs = logits.size(0)
        device = logits.device
        onehot_lsm = to_onehot_lsm(ys, self.vocab_size, self.lsm_prob)
        loss = 0

        for b in range(bs):
            xlen = xlens[b]
            ylen = ylens[b]

            loss_soft = 0
            loss_hard = 0

            align = aligns[b][:xlen].long()

            label_map = self._frame_to_label_mapping(align, xlen, ylen)

            label_exists = (label_map >= 0).unsqueeze(1).to(device)  # (T, 1)

            if self.soft_label_weight > 0:
                # P_LM * log(P_ASR)
                # sum for xlen and vocab
                loss_soft = torch.sum(
                    label_exists
                    * soft_labels[b][label_map]
                    * torch.log_softmax(logits[b][:xlen], dim=-1)
                )
                if self.normalize_length:
                    loss_soft /= label_exists.sum()
            else:
                loss_soft = 0

            if self.soft_label_weight < 1:
                loss_hard = torch.sum(
                    label_exists
                    * onehot_lsm[b][label_map]
                    * torch.log_softmax(logits[b][:xlen], dim=-1)
                )
                if self.normalize_length:
                    loss_hard /= label_exists.sum()
            else:
                loss_hard = 0

            loss -= (
                self.soft_label_weight * loss_soft
                + (1 - self.soft_label_weight) * loss_hard
            )

        if self.normalize_batch:
            loss /= bs

        return loss

    def _frame_to_label_mapping(self, align, xlen, ylen):
        label_map = torch.full((xlen,), -1, dtype=torch.long)
        
        label_id = -1
        left_t, right_t = -1, -1
        for t in range(xlen):
            token_id = align[t]
            if token_id == self.blank_id:
                continue  # label_map[t] = -1
            if t == 0 or token_id != align[t - 1]:  # new token
                if label_id >= 0:
                    if self.position == "left":
                        label_map[left_t] = label_id
                    elif self.position == "mid":
                        mid_t = (left_t + right_t) // 2
                        label_map[mid_t] = label_id
                    elif self.position == "right":
                        label_map[right_t] = label_id
                
                # update `label_id` here
                label_id += 1
                left_t, right_t = t, t

            if self.position == "all":
                label_map[t] = label_id
            right_t = t
        
        if label_id >= 0:
            if self.position == "left":
                label_map[left_t] = label_id
            elif self.position == "mid":
                mid_t = (left_t + right_t) // 2
                label_map[mid_t] = label_id
            elif self.position == "right":
                label_map[right_t] = label_id
        
        assert label_id == (ylen - 1)

        return label_map


class RNNTWordDistillLoss(nn.Module):
    def __init__(
        self, normalize_length=True, normalize_batch=True,
    ):
        super(RNNTWordDistillLoss, self).__init__()

        self.normalize_length = normalize_length
        self.normalize_batch = normalize_batch

    def forward(self, logits, soft_labels, xlens, ylens):
        # logits: (B, T, L + 1, vocab)
        bs = logits.size(0)
        loss = 0

        for b in range(bs):
            xlen = xlens[b]
            ylen = ylens[b]
            soft_label_ext = soft_labels[b, :ylen].unsqueeze(
                0
            )  # (L, vocab) -> (1, L, vocab)

            loss_b = torch.sum(
                soft_label_ext * torch.log_softmax(logits[b, :xlen, :ylen], dim=-1)
            )

            if self.normalize_length:
                loss_b /= xlen * ylen
            loss -= loss_b

        if self.normalize_batch:
            loss /= bs

        return loss


class RNNTAlignDistillLoss(nn.Module):
    def __init__(
        self, normalize_length=True, normalize_batch=True,
    ):
        super(RNNTAlignDistillLoss, self).__init__()

        self.normalize_length = normalize_length
        self.normalize_batch = normalize_batch

    def forward(self, logits, ys, soft_labels, aligns, xlens, ylens):
        # logits: (B, T, L + 1, vocab)
        bs = logits.size(0)
        device = logits.device
        loss = 0

        # TODO: speedup
        for b in range(bs):
            xlen = xlens[b]
            ylen = ylens[b]
            align = aligns[b]

            for u in range(ylen):
                loss_u = torch.sum(
                    soft_labels[b, u]
                    * torch.log_softmax(logits[b, align[u], u], dim=-1),
                    dim=-1,
                )

            if self.normalize_length:
                loss_u /= ylen
            loss -= loss_u

        if self.normalize_batch:
            loss /= bs

        return loss

if __name__ == "__main__":
    # loss = CTCAlignDistillLoss(vocab_size=100, position="left")
    # loss = CTCAlignDistillLoss(vocab_size=100, position="mid")
    # loss = CTCAlignDistillLoss(vocab_size=100, position="right")
    loss = CTCAlignDistillLoss(vocab_size=100, position="all")
    align = [5, 0, 0, 15, 15, 15, 15, 10, 10, 0]
    label_map = loss._frame_to_label_mapping(align, xlen=10, ylen=3)
    print(label_map)
