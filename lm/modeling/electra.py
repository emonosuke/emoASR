import logging
import os
import sys

import torch
import torch.nn as nn

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from asr.modeling.model_utils import make_nopad_mask
from utils.log import get_num_parameters

from lm.modeling.p2w import P2W
from lm.modeling.transformers.configuration_electra import ElectraConfig
from lm.modeling.transformers.modeling_electra import (
    ElectraForMaskedLM,
    ElectraForPreTraining,
)


def sample_temp(logits, temp=1.0):
    if temp == 0.0:
        return torch.argmax(logits)
    probs = torch.softmax(logits / temp, dim=2)  # smoothed softmax
    # sampling from probs
    sample_ids = (
        probs.view(-1, probs.shape[2])
        .multinomial(1, replacement=False)
        .view(probs.shape[0], -1)
    )
    return sample_ids


class ELECTRAModel(nn.Module):
    def __init__(self, params):
        super(ELECTRAModel, self).__init__()

        # Generator
        gconfig = ElectraConfig(
            vocab_size=params.vocab_size,
            hidden_size=params.gen_hidden_size,
            embedding_size=params.gen_embedding_size,
            num_hidden_layers=params.gen_num_layers,
            num_attention_heads=params.gen_num_attention_heads,
            intermediate_size=params.gen_intermediate_size,
            max_position_embeddings=params.max_seq_len,
        )
        self.gmodel = ElectraForMaskedLM(config=gconfig)
        num_params, num_params_trainable = get_num_parameters(self.gmodel)
        logging.info(
            f"ELECTRA: Generator #parameters: {num_params} ({num_params_trainable} trainable)"
        )

        # Discrminator
        dconfig = ElectraConfig(
            vocab_size=params.vocab_size,
            hidden_size=params.disc_hidden_size,
            embedding_size=params.disc_embedding_size,
            num_hidden_layers=params.disc_num_layers,
            num_attention_heads=params.disc_num_attention_heads,
            intermediate_size=params.disc_intermediate_size,
            max_position_embeddings=params.max_seq_len,
        )
        self.dmodel = ElectraForPreTraining(config=dconfig)
        num_params, num_params_trainable = get_num_parameters(self.dmodel)
        logging.info(
            f"ELECTRA: Discriminator #parameters: {num_params} ({num_params_trainable} trainable)"
        )

        self.electra_disc_weight = params.electra_disc_weight

    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        if ylens is None:
            attention_mask = None
        else:
            attention_mask = make_nopad_mask(ylens).float().to(ys.device)
            ys = ys[:, : max(ylens)]  # DataParallel

        gloss, glogits = self.gmodel(ys, attention_mask=attention_mask, labels=labels)

        generated_ids = ys.clone()
        masked_indices = labels.long() != -100
        original_ids = ys.clone()
        original_ids[masked_indices] = labels[masked_indices]
        sample_ids = sample_temp(glogits)  # sampling
        generated_ids[masked_indices] = sample_ids[masked_indices]
        labels_replaced = (generated_ids.long() != original_ids.long()).long()

        dloss, dlogits = self.dmodel(
            generated_ids, attention_mask=attention_mask, labels=labels_replaced
        )

        loss = gloss + self.electra_disc_weight * dloss
        loss_dict = {}

        loss_dict["loss_gen"] = gloss
        loss_dict["loss_disc"] = dloss
        loss_dict["num_replaced"] = labels_replaced.sum().long() / ys.size(0)
        loss_dict["num_masked"] = masked_indices.sum().long() / ys.size(0)

        return loss, loss_dict


class PELECTRAModel(nn.Module):
    """
    Phone-attentive ELECTRA
    """

    def __init__(self, params):
        super(PELECTRAModel, self).__init__()

        # Generator: condictional MLM
        self.gmodel = P2W(
            params,
            cmlm=True,
            encoder_type="transformer",
            decoder_type="bert",
            return_logits=True,
        )
        num_params, num_params_trainable = get_num_parameters(self.gmodel)
        logging.info(
            f"PELECTRA: Generator #parameters: {num_params} ({num_params_trainable} trainable)"
        )

        # Discrminator
        dconfig = ElectraConfig(
            vocab_size=params.vocab_size,
            hidden_size=params.disc_hidden_size,
            embedding_size=params.disc_embedding_size,
            num_hidden_layers=params.disc_num_layers,
            num_attention_heads=params.disc_num_attention_heads,
            intermediate_size=params.disc_intermediate_size,
            max_position_embeddings=params.max_seq_len,
        )
        self.dmodel = ElectraForPreTraining(config=dconfig)
        num_params, num_params_trainable = get_num_parameters(self.dmodel)
        logging.info(
            f"PELECTRA: Discriminator #parameters: {num_params} ({num_params_trainable} trainable)"
        )

        self.electra_disc_weight = params.electra_disc_weight

    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        if ylens is None:
            attention_ymask = None
        else:
            attention_ymask = make_nopad_mask(ylens).float().to(ys.device)
            # DataParallel
            ys = ys[:, : max(ylens)]
            ps = ps[:, : max(plens)]
            labels = labels[:, : max(ylens)]

        gloss, _, glogits = self.gmodel(ys, ylens, labels=labels, ps=ps, plens=plens)

        generated_ids = ys.clone()
        masked_indices = labels.long() != -100
        original_ids = ys.clone()
        original_ids[masked_indices] = labels[masked_indices]
        sample_ids = sample_temp(glogits)  # sampling
        generated_ids[masked_indices] = sample_ids[masked_indices]
        labels_replaced = (generated_ids.long() != original_ids.long()).long()

        dloss, dlogits = self.dmodel(
            generated_ids, attention_mask=attention_ymask, labels=labels_replaced
        )

        loss = gloss + self.electra_disc_weight * dloss
        loss_dict = {}

        loss_dict["loss_gen"] = gloss
        loss_dict["loss_disc"] = dloss
        loss_dict["num_replaced"] = labels_replaced.sum().long() / ys.size(0)
        loss_dict["num_masked"] = masked_indices.sum().long() / ys.size(0)

        return loss, loss_dict
