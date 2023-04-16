import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead


# clone coding

class BertForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids =None,
        attention_mask =None,
        token_type_ids = None,
        mask_pos =None,
        labels =None,
        ):

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos] # 마스크 부분의 representation만을 뽑아냄.

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output) # BertMLMHead

        # Exit early and only return mask logits.
        if self.return_full_softmax: # 모든 단어의 softmax 값들을 가져오는 것이라고 생각
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = [] # 라벨 부분의 값만을 가져온다. 
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output