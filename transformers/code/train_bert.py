from transformers import BertTokenizer
from pathlib import Path
import torch
import csv
from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split
import os
import datetime
import time
from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, accuracy_multilabel, accuracy_thresh, fbeta, roc_auc

import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

DATA_PATH = Path('.')
LABEL_PATH = Path('.')
MODEL_PATH=Path('../models/')
LOG_PATH=Path('../logs/')
MODEL_PATH.mkdir(exist_ok=True)
model_state_dict = None
BERT_PRETRAINED_PATH = None
FINETUNED_PATH = None
LOG_PATH.mkdir(exist_ok=True)
OUTPUT_PATH = MODEL_PATH/'output'
OUTPUT_PATH.mkdir(exist_ok=True)
args = Box({
    "run_text": "multilabel topic classification with freezable layers",
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "review_classification_lib",
    "no_cuda": False,
    "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 20,
    "warmup_proportion": 0.0,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "fp16_opt_level": "O1",
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "warmup_steps": 500,
    "logging_steps": 50,
    "eval_all_checkpoints": True,
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "seed": 42,
    "loss_scale": 128,
    "task_name": 'topics',
    "model_name": 'bert-base-cased',
    "model_type": 'bert'
})

import logging

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()
logger.info(args)
if torch.cuda.device_count() > 1:
    args.multi_gpu = True
else:
    args.multi_gpu = False

label_cols = ["functionality", "range_anxiety", "availability", "cost", "ui", "location", "service_time", "dealership"]

databunch = BertDataBunch(args['data_dir'], LABEL_PATH, args.model_name, train_file='train_final.csv', val_file='valid_final.csv',
                          test_data='test_final.csv',
                          text_col="review", label_col=label_cols,
                          batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'], 
                          multi_gpu=args.multi_gpu, multi_label=True, model_type=args.model_type)

databunch.train_dl.dataset[0][3]
num_labels = len(databunch.labels)
print(num_labels)
metrics = []
metrics.append({'name': 'accuracy', 'function': accuracy})
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})
print(device)

learner = BertLearner.from_pretrained_model(databunch, args.model_name, metrics=metrics, 
                                            device=device, logger=logger, output_dir=args.output_dir, 
                                            finetuned_wgts_path=FINETUNED_PATH, warmup_steps=args.warmup_steps,
                                            multi_gpu=args.multi_gpu, is_fp16=args.fp16, 
                                            multi_label=True, logging_steps=0)


start_fit = time.time()
learner.fit(args.num_train_epochs, args.learning_rate, validate=True)
end_fit = time.time()
learner.validate()
learner.save_model()
start_predict = time.time()
predictions = learner.predict_batch(list(pd.read_csv('test_final.csv')['review'].values))
end_predict = time.time()

print("Training time : {}".format(end_fit - start_fit))
print("Prediction time : {}".format(end_predict - start_predict))
