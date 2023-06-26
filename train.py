import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
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


parser = argparse.ArgumentParser()
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=6)
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--embedding_size', type=int, default=128,
                    help='Embedding dimension for the spatial coordinates')

# Size of each batch parameter
parser.add_argument('--batch_size', type=int, default=1024,
                    help='minibatch size')
# Length of sequence to be considered parameter
parser.add_argument('--seq_length', type=int, default=350,
                    help='RNN sequence length')
parser.add_argument('--pred_length', type=int, default=50,
                    help='prediction length')
parser.add_argument('--multiply', type=int, default=5,
                    help='prediction length')
parser.add_argument('--window_width', type=int, default=5,
                    help='prediction length')
# Number of epochs parameter
parser.add_argument('--num_epochs', type=int, default=200,
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
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout probability')
# Dimension of the embeddings parameter
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
parser.add_argument('--lambda_param', type=float, default=0.00005,
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
parser.add_argument('--converge_coeff', type=float, default=0.1,
                    help='Help convergence')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data( user2info,user2traj,):
    x = []
    y = []
    traj_all = []
    min_lon = 1000
    max_lon = 0
    min_lat = 1000
    max_lat = 0
    for val in user2traj.values():
        '''
        traj = np.array(val['traj'])
        if(len(traj)<1):
            continue
        min_lon = np.min([traj[:,2].min(),min_lon])
        max_lon = np.max([traj[:,2].max(),max_lon])
        min_lat = np.min([traj[:,3].min(),min_lat])
        max_lat = np.max([traj[:,3].max(),max_lat])
        '''



    for user in user2traj:
        traj = user2traj[user]['traj']
        traj = np.array(traj)
        if len(traj) > args.seq_length + args.pred_length + args.multiply:
            for mul in range(args.multiply):
                label = np.zeros((3,2))
                label[0, user2info[user]['gender']] = 1
                label[1, user2info[user]['income']] = 1
                if user2info[user]['edu'] == 0:
                    label[2, 0] = 1
                else:
                    label[2, 1] = 1

                for i in range(int(len(traj)/(args.pred_length+args.seq_length))-1):
                    traj_x = traj[i*(args.pred_length+args.seq_length)+mul*args.window_width:(i+1)*(args.pred_length+args.seq_length)+mul*args.window_width, 2:]
                    #print(len(traj_x))
                    #print(len(traj_x[0]))
                    min_lon = np.min(traj_x[:, 0])
                    max_lon = np.max(traj_x[:, 0])
                    min_lat = np.min(traj_x[:, 1])
                    max_lat = np.max(traj_x[:, 1])
                    lon_dis = max_lon-min_lon if (max_lon-min_lon)>1e-5 else 1
                    lat_dis = max_lat-min_lat if (max_lat-min_lat)>1e-5 else 1
                    traj_x[:, 0] = (traj_x[:, 0] - min_lon)/lon_dis
                    traj_x[:, 1] = (traj_x[:, 1] - min_lat)/lat_dis

                    x.append(traj_x)
                    y.append(label)
    x = np.array(x)
    y = np.array(y)

    return x, y


# Load data
'''
with open('user2info_level.json', 'r') as f:
    user2info = json.load(f,)

with open('user2traj.json', 'r') as f:
    user2traj = json.load(f,)



x,y = load_data(user2info, user2traj)
y = np.argmax(y,axis=-1)
permutation = np.random.permutation(len(x))
print(len(x))
x = x[permutation]
y = y[permutation]
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x[:int(0.8*x.shape[0])]), torch.from_numpy(y[:int(0.8*x.shape[0])]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_loader = pickle.load()
valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x[int(0.8*x.shape[0]):]), torch.from_numpy(y[int(0.8*x.shape[0]):]))
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
'''
train_loader = pickle.load(open('./sep_train_test/train_Ten.pkl','rb'))
valid_loader = pickle.load(open('./sep_train_test/test_Ten0.8-0.9.pkl','rb'))

model = model.SocialModel(args).to(device)
Loss_CE = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
Loss_MSE = nn.MSELoss()
record_train = {
    'loss':[],
    'gender':[],
    'income':[],
    'edu':[]
}

record_valid = {
    'loss':[],
    'gender':[],
    'income':[],
    'edu':[]
}

valid_loss_best = 1e5
model.load('logs/model_last_Ten.pkl')
for epoch in range(args.num_epochs):
    start_time = time.time()
    print("Epoch: %d"%epoch)
    total_train_step = 0
    record_loss = []
    record_gender = []
    record_income = []
    record_edu = []
    record_traj = []
    record_regu = []
    n=0
    model.train()

    for x,y in train_loader:
        x = x.to(torch.float32).to(device)
        x_i = x[:,:args.seq_length,:]
        x_o = x[:,args.seq_length:,:]
        y = y.to(torch.long).to(device)
        y_pred, x_pred = model(x_i)
        gender_pred = y_pred[:, 0,:]
        income_pred = y_pred[:, 1,:]
        edu_pred = y_pred[:, 2,:]
        gender = y[:, 0]
        income = y[:, 1]
        edu = y[:, 2]
        loss_gender = Loss_CE(gender_pred,gender)
        loss_income = Loss_CE(income_pred,income)
        loss_edu = Loss_CE(edu_pred,edu)
        loss_traj = Loss_MSE(x_pred,x_o)

        loss = args.converge_coeff * (loss_gender + loss_income + loss_edu) + loss_traj  
    
        loss = loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record_loss.append( loss.cpu().item() )
        total_train_step += 1

        pred_index = torch.argmax(y_pred,dim=-1)
        record_gender.append( ((pred_index[:,0]==y[:,0]).sum()/y.shape[0]).item() )
        record_income.append( ((pred_index[:,1]==y[:,1]).sum()/y.shape[0]).item() )
        record_edu.append( ((pred_index[:,2]==y[:,2]).sum()/y.shape[0]).item() )
        record_traj.append(loss_traj.cpu().item())
        #record_regu.append(regularization_loss.cpu().item())
        '''
        if n%50==0:
            print(np.mean(record_gender),np.mean(record_income),np.mean(record_edu))
            record_gender = []
            record_income = []
            record_edu = []
        n+=1
        '''
    print('train loss : ',np.mean(record_loss),np.mean(record_gender),np.mean(record_income),np.mean(record_edu),np.mean(record_traj))
    record_train['loss'].append(np.mean(record_loss))
    record_train['gender'].append(np.mean(record_gender))
    record_train['income'].append(np.mean(record_income))
    record_train['edu'].append(np.mean(record_edu))
    #record_train['traj'].append(np.mean(record_loss))
    record_loss = []
    record_gender = []
    record_income = []
    record_edu = []
    record_traj = []

    n=0
    model.eval()

    for x,y in valid_loader:
        
        x = x.to(torch.float32).to(device)
        x_i = x[:,:args.seq_length,:]
        x_o = x[:,args.seq_length:,:]
        y = y.to(torch.long).to(device)
        y_pred, x_pred = model(x_i)
        gender_pred = y_pred[:, 0,:]
        income_pred = y_pred[:, 1,:]
        edu_pred = y_pred[:, 2,:]
        gender = y[:, 0]
        income = y[:, 1]
        edu = y[:, 2]
        loss_gender = Loss_CE(gender_pred,gender)
        loss_income = Loss_CE(income_pred,income)
        loss_edu = Loss_CE(edu_pred,edu)
        loss_traj = Loss_MSE(x_pred,x_o)
        loss = args.converge_coeff * (loss_gender + loss_income + loss_edu) + loss_traj        
        record_loss.append( loss.cpu().item() )
        total_train_step += 1

        pred_index = torch.argmax(y_pred,dim=-1)
        record_gender.append( ((pred_index[:,0]==y[:,0]).sum()/y.shape[0]).item() )
        record_income.append( ((pred_index[:,1]==y[:,1]).sum()/y.shape[0]).item() )
        record_edu.append( ((pred_index[:,2]==y[:,2]).sum()/y.shape[0]).item() )
        record_traj.append(loss_traj.cpu().item())

    print('valid loss : ',np.mean(record_loss),np.mean(record_gender),np.mean(record_income),np.mean(record_edu),np.mean(record_traj))
    record_valid['loss'].append(np.mean(record_loss))
    record_valid['gender'].append(np.mean(record_gender))
    record_valid['income'].append(np.mean(record_income))
    record_valid['edu'].append(np.mean(record_edu))
    #record_valid['traj'].append(np.mean(record_traj))

    with open('logs/log.pkl','wb') as f:
        pickle.dump([record_train,record_valid],f)

    if np.mean(record_loss)<valid_loss_best:
        valid_loss_best = np.mean(record_loss).item()
        model.save('logs/model_Ten.pkl')

    if epoch%20==0:
        model.save(f'logs/model_epoch{epoch}_Ten.pkl')
model.save('logs/model_last_Ten.pkl')    

'''
    model.eval()
    total_valid_step = 0
    l_sum, n = 0.0, 0

    
    for x,y in train_loader:
        x = x.to(torch.float32).to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = Loss(y_pred, y)
        l_sum += loss.cpu().item() *y.shape[0]
        n += y.shape[0]
        total_train_step += 1
        gender_pred = torch.argmax(y_pred[:, 0,:], dim=1)
        income_pred = torch.argmax(y_pred[:, 1,:], dim=1)
        edu_pred = torch.argmax(y_pred[:, 2,:], dim=1)
        gender = torch.argmax(y[:, 0,:], dim=1)
        income = torch.argmax(y[:, 1,:], dim=1)
        edu = torch.argmax(y[:, 2,:], dim=1)
        # print(edu_pred,income_pred,gender_pred)
        acc_gender = (gender==gender_pred).float().sum()
        acc_income = (income==income_pred).float().sum()
        acc_edu = (edu==edu_pred).float().sum()
    '''



    




