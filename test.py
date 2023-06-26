import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import setproctitle
setproctitle.setproctitle("Metaworld@wjw")
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import argparse
import time
import pickle
import pandas
parser = argparse.ArgumentParser()
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=6)
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
# Size of each batch parameter
parser.add_argument('--batch_size', type=int, default=1024,
                    help='minibatch size')
# Length of sequence to be considered parameter
parser.add_argument('--seq_length', type=int, default=350,
                    help='RNN sequence length')
parser.add_argument('--pred_length', type=int, default=50,
                    help='prediction length')
# Number of epochs parameter
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs')
# Frequency at which the model should be saved parameter
parser.add_argument('--save_every', type=int, default=400,
                    help='save frequency')
# TODO: (resolve) Clipping gradients for now. No idea whether we should
# Gradient value at which it should be clipped
parser.add_argument('--grad_clip', type=float, default=10.,
                    help='clip gradients at this value')
# Learning rate parameter
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
# Decay rate for the learning rate parameter
parser.add_argument('--decay_rate', type=float, default=0.95,
                    help='decay rate for rmsprop')
# Dropout not implemented.
# Dropout probability parameter
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout probability')
# Dimension of the embeddings parameter
parser.add_argument('--embedding_size', type=int, default=128,
                    help='Embedding dimension for the spatial coordinates')
# Size of neighborhood to be considered parameter
parser.add_argument('--neighborhood_size', type=int, default=32,
                    help='Neighborhood size to be considered for social grid')
# Size of the social grid parameter
parser.add_argument('--grid_size', type=int, default=4,
                    help='Grid size of the social grid')
# Maximum number of pedestrians to be considered
parser.add_argument('--maxNumPeds', type=int, default=1000,
                    help='Maximum Number of Pedestrians')

# Lambda regularization parameter (L2)
parser.add_argument('--lambda_param', type=float, default=0.0005,
                    help='L2 regularization parameter')
# Cuda parameter
parser.add_argument('--use_cuda', action="store_true", default=True,
                    help='Use GPU or not')
# GRU parameter
parser.add_argument('--gru', action="store_true", default=True,
                    help='True : GRU cell, False: LSTM cell')
# drive option
parser.add_argument('--drive', action="store_true", default=False,
                    help='Use Google drive or not')
# number of validation will be used
parser.add_argument('--num_validation', type=int, default=1,
                    help='Total number of validation dataset for validate accuracy')
# frequency of validation
parser.add_argument('--freq_validation', type=int, default=1,
                    help='Frequency number(epoch) of validation using validation data')
# frequency of optimazer learning decay
parser.add_argument('--freq_optimizer', type=int, default=8,
                    help='Frequency number(epoch) of learning decay for optimizer')
# store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
parser.add_argument('--grid', action="store_true", default=True,
                    help='Whether store grids and use further epoch')
parser.add_argument('--return_embedding', action="store_true", default=False,
                    help='Whether to return embedding for visualization')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

valid_loader = pickle.load(open('./sep_train_test/test0.8-0.9.pkl','rb'))

model = model.SocialModel_Dist(args).to(device)

Loss_MSE = nn.MSELoss()

model.load('logs/model_distan_epoch20_Ten.pkl')


gender_gt_list = []
income_gt_list = []
education_gt_list = []
gender_pred_list = []
income_pred_list = []
education_pred_list = []
mse_loss_list = []

for x,y in valid_loader:
    x = x.to(torch.float32).to(device)
    x_i = x[:,:args.seq_length,:]
    x_o = x[:,args.seq_length:,:]
    y = y.to(torch.long).to(device)

    y_pred, x_pred, dis = model(x_i)
    #y_pred, x_pred = model(x_i)

    y_pred = torch.argmax(y_pred,dim=-1)
    x_pred = x_pred.cpu().detach().numpy()
    x_o = x_o.cpu().detach().numpy()
    loss_traj = np.mean((x_pred - x_o) ** 2, axis=(1, 2))

    gender_gt_list.append(y[:,0].cpu().detach().numpy())
    income_gt_list.append(y[:,1].cpu().detach().numpy())
    education_gt_list.append(y[:,2].cpu().detach().numpy())

    gender_pred_list.append(y_pred[:,0].cpu().detach().numpy())
    income_pred_list.append(y_pred[:,1].cpu().detach().numpy())
    education_pred_list.append(y_pred[:,2].cpu().detach().numpy())
    mse_loss_list.append(loss_traj)
    #y_pred, x_pred, dis_item = model(x_i)
#print(gender_gt_list[0])
#print(gender_gt_list[1])
gender_gt_array = np.concatenate(gender_gt_list, axis=0)
income_gt_array = np.concatenate(income_gt_list, axis=0)
education_gt_array = np.concatenate(education_gt_list, axis=0)
gender_pred_array = np.concatenate(gender_pred_list, axis=0)
income_pred_array = np.concatenate(income_pred_list, axis=0)
education_pred_array = np.concatenate(education_pred_list, axis=0)
mse_loss_array = np.concatenate(mse_loss_list, axis=0)
'''
print(gender_gt_array.shape)
print(income_gt_array.shape)
print(education_gt_array.shape)

print(gender_pred_array.shape)
print(income_pred_array.shape)
print(education_pred_array.shape)

print(mse_loss_array.shape)
'''
data = {
    'gender_gt': gender_gt_array,
    'income_gt': income_gt_array,
    'education_gt': education_gt_array,
    'gender_pred': gender_pred_array,
    'income_pred': income_pred_array,
    'education_pred': education_pred_array,
    'MSE_Loss': mse_loss_array
}

df = pd.DataFrame(data)
df.to_csv('result_dis0.8-0.9.csv', index=False)

