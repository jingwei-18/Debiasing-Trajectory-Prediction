import torch
import torch.nn as nn
import numpy as np


class SocialModel_no(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel_no, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.pred_length=args.pred_length
        self.gru = args.gru
        self.return_embedding = args.return_embedding
    
        # The LSTM cell
        #self.cell = nn.LSTM(self.embedding_size, self.rnn_size,num_layers = 2,batch_first = True)

        if self.gru:
            self.RNN = nn.GRU(self.embedding_size, self.rnn_size,num_layers=3, batch_first = True)
            self.DeRNN = nn.GRU(self.embedding_size, self.rnn_size,num_layers=3, batch_first = True)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.output_embedding_layer = nn.Linear(self.embedding_size, self.input_size)
        # Linear layer to map the hidden state of LSTM to output
        #self.gender_output_layer = nn.Linear(self.rnn_size, 2)
        #self.income_output_layer = nn.Linear(self.rnn_size, 2)
        #self.edu_output_layer = nn.Linear(self.rnn_size, 2)
        #self.RNN_linear = nn.Linear(self.rnn_size, self.rnn_size*3)
        #self.grouping_linear = nn.Linear(self.rnn_size*3, self.rnn_size)
        self.Decode_linear = nn.Linear(self.rnn_size, self.embedding_size)
        # ReLU and dropout unit
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        batch = x.size(0)
        seq_length = x.size(1)
        input_size = x.size(2)
        # print('init',x)
        x = self.dropout(self.relu(self.input_embedding_layer(x)))
        #print(x.shape)
        # print("embedidng",x)
        #x = x.reshape(batch*seq_length, self.embedding_size)
        #x = self.cell(x)[0]
        x_outputs = torch.zeros(self.pred_length, batch, input_size)
        output, hidden  = self.RNN(x)
        #print(hidden.shape)
        #traj_emb = hidden.squeeze(0)

        #traj_emb = self.RNN_linear(traj_emb)
        #traj_emb_div = traj_emb.reshape(traj_emb.shape[0],self.rnn_size,3)
        #traj_emb = traj_emb.reshape(batch,self.rnn_size,3)
        #gender = self.gender_output_layer(hidden[0,:,:].squeeze(0))
        #income = self.income_output_layer(hidden[1,:,:].squeeze(0))
        #edu = self.edu_output_layer(hidden[2,:,:].squeeze(0))

        #traj_emb = self.relu(self.grouping_linear(traj_emb))
        decoder_input = x[:,-1,:].unsqueeze(1)
        decoder_hidden = hidden
        for pred_i in range(self.pred_length):
            #print(decoder_input.shape)
            #print(decoder_hidden.shape)
            decoder_input, decoder_hidden = self.DeRNN(decoder_input, decoder_hidden)
            decoder_input = self.relu(decoder_input)
            decoder_input = self.Decode_linear(decoder_input)
            #print(decoder_input.shape)
            x_outputs[pred_i] = self.output_embedding_layer(decoder_input.squeeze(1))

        # print('lstm',x)
        # print(x.shape)
        #x = x.reshape(batch, seq_length, self.rnn_size)
        #x = torch.sum(x, dim=1)
        # print('sum',x)
        #x = self.softmax(x)
        #x = self.dropout(x)
        # print(gender, income, edu)
        #y = torch.stack((gender, income, edu), dim=1)
        x_outputs = x_outputs.permute(1,0,2).cuda()
        if self.return_embedding:
            return x_outputs, hidden
        else:
            return x_outputs

    def save(self,dir):
        state={
        'model': self.state_dict(),
        }
        torch.save(state, dir)

    def load(self,dir):
        data = torch.load(dir)
        state_dict = data['model']
        self.load_state_dict(state_dict)


class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.pred_length=args.pred_length
        self.gru = args.gru
        self.return_embedding = args.return_embedding
        
    
        # The LSTM cell
        #self.cell = nn.LSTM(self.embedding_size, self.rnn_size,num_layers = 2,batch_first = True)

        if self.gru:
            self.RNN = nn.GRU(self.embedding_size, self.rnn_size,num_layers=3, batch_first = True)
            self.DeRNN = nn.GRU(self.embedding_size, self.rnn_size,num_layers=3, batch_first = True)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.output_embedding_layer = nn.Linear(self.embedding_size, self.input_size)
        # Linear layer to map the hidden state of LSTM to output
        self.gender_output_layer = nn.Linear(self.rnn_size, 2)
        self.income_output_layer = nn.Linear(self.rnn_size, 2)
        self.edu_output_layer = nn.Linear(self.rnn_size, 2)
        #self.RNN_linear = nn.Linear(self.rnn_size, self.rnn_size*3)
        #self.grouping_linear = nn.Linear(self.rnn_size*3, self.rnn_size)
        self.Decode_linear = nn.Linear(self.rnn_size, self.embedding_size)
        # ReLU and dropout unit
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        batch = x.size(0)
        seq_length = x.size(1)
        input_size = x.size(2)
        # print('init',x)
        x = self.dropout(self.relu(self.input_embedding_layer(x)))
        #print(x.shape)
        # print("embedidng",x)
        #x = x.reshape(batch*seq_length, self.embedding_size)
        #x = self.cell(x)[0]
        x_outputs = torch.zeros(self.pred_length, batch, input_size)
        output, hidden  = self.RNN(x)
        #print(hidden.shape)
        #traj_emb = hidden.squeeze(0)

        #traj_emb = self.RNN_linear(traj_emb)
        #traj_emb_div = traj_emb.reshape(traj_emb.shape[0],self.rnn_size,3)
        #traj_emb = traj_emb.reshape(batch,self.rnn_size,3)
        gender = self.gender_output_layer(hidden[0,:,:].squeeze(0))
        income = self.income_output_layer(hidden[1,:,:].squeeze(0))
        edu = self.edu_output_layer(hidden[2,:,:].squeeze(0))

        #traj_emb = self.relu(self.grouping_linear(traj_emb))
        decoder_input = x[:,-1,:].unsqueeze(1)
        decoder_hidden = hidden
        for pred_i in range(self.pred_length):
            #print(decoder_input.shape)
            #print(decoder_hidden.shape)
            decoder_input, decoder_hidden = self.DeRNN(decoder_input, decoder_hidden)
            decoder_input = self.relu(decoder_input)
            decoder_input = self.Decode_linear(decoder_input)
            #print(decoder_input.shape)
            x_outputs[pred_i] = self.output_embedding_layer(decoder_input.squeeze(1))

        # print('lstm',x)
        # print(x.shape)
        #x = x.reshape(batch, seq_length, self.rnn_size)
        #x = torch.sum(x, dim=1)
        # print('sum',x)
        #x = self.softmax(x)
        #x = self.dropout(x)
        # print(gender, income, edu)
        y = torch.stack((gender, income, edu), dim=1)
        x_outputs = x_outputs.permute(1,0,2).cuda()
        
        if self.return_embedding:
            return y, x_outputs, hidden
        else:
            return y, x_outputs

    def save(self,dir):
        state={
        'model': self.state_dict(),
        }
        torch.save(state, dir)

    def load(self,dir):
        data = torch.load(dir)
        state_dict = data['model']
        self.load_state_dict(state_dict)
            

class SocialModel_Dist(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel_Dist, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.pred_length=args.pred_length
        self.gru = args.gru
        self.return_embedding = args.return_embedding
        
    
        # The LSTM cell
        #self.cell = nn.LSTM(self.embedding_size, self.rnn_size,num_layers = 2,batch_first = True)

        if self.gru:
            self.RNN = nn.GRU(self.embedding_size, self.rnn_size,num_layers=3, batch_first = True)
            self.DeRNN = nn.GRU(self.embedding_size, self.rnn_size,num_layers=3, batch_first = True)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.output_embedding_layer = nn.Linear(self.embedding_size, self.input_size)
        # Linear layer to map the hidden state of LSTM to output
        self.gender_output_layer = nn.Linear(self.rnn_size, 2)
        self.income_output_layer = nn.Linear(self.rnn_size, 2)
        self.edu_output_layer = nn.Linear(self.rnn_size, 2)
        self.RNN_linear = nn.Linear(self.rnn_size, self.rnn_size*3)

        # ReLU and dropout unit
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        batch = x.size(0)
        seq_length = x.size(1)
        input_size = x.size(2)
        # print('init',x)
        x = self.dropout(self.relu(self.input_embedding_layer(x)))
        
        # print("embedidng",x)
        #x = x.reshape(batch*seq_length, self.embedding_size)
        #x = self.cell(x)[0]
        x_outputs = torch.zeros(self.pred_length, batch, input_size)
        output, hidden  = self.RNN(x)
        #print(hidden.shape)

        gender = self.gender_output_layer(hidden[0,:,:])
        income = self.income_output_layer(hidden[1,:,:])
        edu = self.edu_output_layer(hidden[2,:,:])
        #print(gender.shape)
        output_i = []
        output_i.append(torch.matmul(hidden[0,:,:],self.income_output_layer.weight.detach().T)+self.income_output_layer.bias.detach()[None,:])
        output_i.append(torch.matmul(hidden[0,:,:],self.edu_output_layer.weight.detach().T)+self.edu_output_layer.bias.detach()[None,:])

        output_i.append(torch.matmul(hidden[1,:,:],self.gender_output_layer.weight.detach().T)+self.gender_output_layer.bias.detach()[None,:])
        output_i.append(torch.matmul(hidden[1,:,:],self.edu_output_layer.weight.detach().T)+self.edu_output_layer.bias.detach()[None,:])

        output_i.append(torch.matmul(hidden[2,:,:],self.gender_output_layer.weight.detach().T)+self.gender_output_layer.bias.detach()[None,:])
        output_i.append(torch.matmul(hidden[2,:,:],self.income_output_layer.weight.detach().T)+self.income_output_layer.bias.detach()[None,:])
        #print(hidden.size)
        decoder_input = x[:,-1,:].unsqueeze(1)
        decoder_hidden = hidden
        for pred_i in range(self.pred_length):
            decoder_input, decoder_hidden = self.DeRNN(decoder_input, decoder_hidden)
            decoder_input = self.relu(decoder_input)
            x_outputs[pred_i] = self.output_embedding_layer(decoder_input.squeeze(1))

        #output_i.append(torch.matmul(traj_emb[:,:,2],self.gender_output_layer.weight.detach().T)+self.gender_output_layer.bias.detach()[None,:])

        # print(gender, income, edu)
        y = torch.stack((gender, income, edu), dim=1)
        x_outputs = x_outputs.permute(1,0,2).cuda()
        if self.return_embedding:
            return y, x_outputs,output_i, hidden
        else:
            return y, x_outputs,output_i

    def save(self,dir):
        state={
        'model': self.state_dict(),
        }
        torch.save(state, dir)

    def load(self,dir):
        data = torch.load(dir)
        state_dict = data['model']
        self.load_state_dict(state_dict)
 

