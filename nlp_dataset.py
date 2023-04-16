import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score
import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import sys
from io import StringIO
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import emoji
from soynlp.normalizer import repeat_normalize
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers import DataProcessor, InputExample
import os

from dataclasses import dataclass
from transformers.data.processors.utils import InputFeatures
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b
        
def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]


def tokenize_multipart_input(
    input_text_list, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    support_labels=None,
):
    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    if prompt: 

        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 
        } 

        template_list = template.split('*') # Get variable list in the template
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

        for _ , part in enumerate(template_list):
            new_tokens = []
            if part in special_token_mapping:
                new_tokens.append(special_token_mapping[part])

            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id]) 

            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
                
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            #! 진짜 모델에 입력 될 값들
            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

    else:
        pass # ...

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        pass
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        #logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))    

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]

        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}

    result['token_type_ids'] = token_type_ids

    if True: #! ALWAYS PROMPT
        result['mask_pos'] = mask_pos

    result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
    result['attention_mask']= torch.tensor(result['attention_mask'], dtype= torch.long)
    result['token_type_ids']= torch.tensor(result['token_type_ids'], dtype= torch.long)
    result['mask_pos']= torch.tensor(result['mask_pos'], dtype= torch.long)

    return result


class DARTProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "nlp_train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "nlp_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "nlp_valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "nlp_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["5", "4", "3", "2", "1", "0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class FewShotDataset(torch.utils.data.Dataset):

    def __init__(self, args, tokenizer, mode="train", use_demo=False):
        self.args = args
        self.processor = args.processor #todo see processor
        self.tokenizer = tokenizer
        self.mode = mode

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)

        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                self.label_to_word[key] = tokenizer.convert_tokens_to_ids(self.label_to_word[key])
            logger.info("Label {} to word {} ({})".format(key, tokenizer.convert_ids_to_tokens(self.label_to_word[key]), self.label_to_word[key]))
            print("Label {} to word {} ({})".format(key, tokenizer.convert_ids_to_tokens(self.label_to_word[key]), self.label_to_word[key]))
            # regression 안쓸거다
            self.label_word_list = [self.label_to_word[label] for label in self.label_list]

        else:
            self.label_to_word = None
            self.label_word_list = None

        self.num_sample = 1 # mode == train, not use demo
        
        self.support_examples = self.processor.get_train_examples(args.data_dir)

        if mode == "dev":
            self.query_examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            self.query_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.query_examples = self.support_examples

         # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        # Prepare examples (especially for using demonstrations)
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                # Using demonstrations without filtering
                context_indices = [support_idx for support_idx in support_indices
                            if support_idx != query_idx or mode != "train"]

                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if mode != "train":
            self.features = []
            _ = 0
            for query_idx, context_indices, bootstrap_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]

                self.features.append(self.convert_fn(
                    example=example,
                    label_list=self.label_list,
                    template=self.args.template,
                    prompt=args.prompt,
                    label_word_list=self.label_word_list,
                    verbose=True if _ == 0 else False,
                ))

                _ += 1
        else:
            self.features = None
    def __len__(self):
        return self.size

    def get_labels(self):
        return self.label_list

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]

            features = self.convert_fn(
                example=example,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template = self.args.template,
                label_word_list=self.label_word_list,
                verbose=False,
            )
        else:
            features = self.features[i]
            
        return features

    def convert_fn(
        self,
        example,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length    

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        else:
            example_label = label_map[example.label]

        # Prepare other features
        if not use_demo:
            # No using demonstrations
            inputs = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                truncate_head=self.args.truncate_head,
            )
            
            inputs['labels'] = example_label
            features = inputs

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            # logger.info("text: %s" % self.tokenizer.decode(features.input_ids)) # 이거 실행하면 너무 길어짐.

        return features