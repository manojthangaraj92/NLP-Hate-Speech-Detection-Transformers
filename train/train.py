import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import random, tqdm, sys, math, gzip
# from torchtext import data, datasets, vocab
from torchtext.legacy import data, datasets, vocab
import numpy as np
from model import CTransformer
import argparse
import json
import os
import pickle
import sys
#import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import spacy
from spacy.cli.download import download
download('en_core_web_sm')
spacy.load('en_core_web_sm')

def build_vocab(file_path):
    #Reproducing same results
    SEED = 2019

    #Torch
    torch.manual_seed(SEED)

    TEXT = data.Field(tokenize='spacy',lower=True, include_lengths=True, batch_first=True)
    LABEL = data.LabelField(batch_first=True)
    fields = [(None,None),('tweet', TEXT),('label', LABEL)]

    training_data=data.TabularDataset(path = file_path,format = 'csv',fields = fields,skip_header = True)
    train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))
    TEXT.build_vocab(train_data,min_freq=3, vectors = "glove.6B.100d")  
    LABEL.build_vocab(train_data)  

    #check whether cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    #set batch size
    BATCH_SIZE = 64

    #Load an iterator
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = BATCH_SIZE,
        sort_key = lambda x: len(x.tweet),
        sort_within_batch=True,
        device = device)
    len_text_vocab = len(TEXT.vocab)
    word_dict = TEXT.vocab.stoi
    return train_iterator, valid_iterator, len_text_vocab, word_dict
   

def train(train_loader, test_loader, num_epoch, opt, model):
    seen = 0
    #initialize every epoch 
    epoch_loss = 0
    total, correction= 0.0, 0.0
    #epoch_acc = 0
    for e in range(num_epoch):
        print(f'\n epoch {e}')
        model.train(True)
        for batch in tqdm.tqdm(train_loader):
            opt.zero_grad()
            input = batch.tweet[0]
            label = batch.label

            #if input.size(0) > 512:
                #input = input[:, :512]
            out = model(input)
            output = out.argmax(dim=1)
            loss = F.nll_loss(out, label)
            loss.backward()
            opt.step()
            seen += input.size(0)
            #loss and accuracy
            total += float(input.size(0))
            correction += float((label == output).sum().item())
            epoch_loss += loss.item()
            print('classification/train-loss', float(loss.item()), seen)
        accuracy = correction / total
        print(f'-- {"test validation"} accuracy {accuracy*100}')    
    
        with torch.no_grad():

            model.train(False)
            tot, cor= 0.0, 0.0

            for batch in tqdm.tqdm(test_loader):

                input = batch.tweet[0]
                label = batch.label
                out = model(input).argmax(dim=1)

                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor / tot
            print(f'-- {"test validation"} accuracy {acc*100}')
    torch.save(model.state_dict(), 'saved_weights.pt')
    print("The model is saved")
    pass



if __name__=='__main__':
    #we will sent model parameters and training parameters are sent as arguments when the script is executed.
    
    parser = argparse.ArgumentParser()
    
    #parser.add_argument("-e",
    #                    dest="num_epochs",
    #                    help="Number of epochs.",
    #                    default=10, type=int)
    parser.add_argument('--epochs', type=int, default=2)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    #parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        #help="Tensorboard logging directory",
                        #default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)
    
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = parser.parse_args()
    
    print('ARGS', args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    print(os.listdir())
    

    #path=r's3://sagemaker-us-east-1-289387546977/sagemaker/hatespeech_twitter/'
    print(os.getcwd())
    path = r'/opt/ml/code/df.csv'
    train_iterator, valid_iterator, len_text_vocab, word_dict=build_vocab(path)
    
    NUM_CLS = 2

    # create the model
    model = CTransformer(embedding=args.embedding_size, 
                        heads=args.num_heads, 
                        depth=args.depth, 
                        seq_length=args.max_length, 
                        num_tokens=args.vocab_size, 
                        num_classes=NUM_CLS, 
                        max_pool=True)
    if torch.cuda.is_available():
        model.cuda()
    opt = torch.optim.Adam(lr=args.lr, params=model.parameters())
    train(train_iterator, valid_iterator, 10, opt, model)
    
    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_size,
            'heads': args.num_heads,
            'vocab_size': args.vocab_size,
            'depth':args.depth,
            'seq_length':args.max_length
        }
        torch.save(model_info, f)
    
    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

