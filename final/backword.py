import pandas as pd
import csv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from IPython.core.display import HTML
from matplotlib import animation
from matplotlib.colors import ListedColormap

with open('origin.csv') as f:
    df_origin = pd.read_csv(f)
with open('processed.csv') as f:
    df_processed = pd.read_csv(f)

#code reference: https://github.com/rasbt/deeplearning-models
#code reference: http://zhaoxuhui.top/blog/2018/09/11/PyTorchNote2.html
#code reference: https://zhuanlan.zhihu.com/p/55600212
df_origin_macro = df_origin[df_origin['wavenumber'] <= 2e-2]
df_origin_micro = df_origin[df_origin['wavenumber'] > 2e-2]
df_processed_macro = df_processed[df_processed['wavenumber'] == 0]
df_processed_micro = df_processed[df_processed['wavenumber'] != 0]

def prediction_visualization():
    fig1, ax1 = plt.subplots(6,1,figsize=(16,16))
    ax1[0].plot(range(1,len(df_origin)+1),df_origin.iloc[:,0], 'o')
    ax1[0].set_ylabel('value', fontsize=20)
    ax1[0].legend(['(K/u)^L'], fontsize=20)

    ax1[1].plot(range(1,len(df_origin)+1),df_origin.iloc[:,1], 'o')
    ax1[1].set_ylabel('value', fontsize=20)
    ax1[1].legend(['(K/u)^M'], fontsize=20)

    ax1[2].plot(range(1,len(df_origin)+1),df_origin.iloc[:,2], 'o')
    ax1[2].set_ylabel('value', fontsize=20)
    ax1[2].legend(['Jm^L'], fontsize=20)

    ax1[3].plot(range(1,len(df_origin)+1),df_origin.iloc[:,3], 'o')
    ax1[3].set_ylabel('value', fontsize=20)
    ax1[3].legend(['Jm^M'], fontsize=20)

    ax1[4].plot(range(1,len(df_origin)+1),df_origin.iloc[:,4], 'o')
    ax1[4].set_ylabel('value', fontsize=20)
    ax1[4].legend(['u^L/u^M'], fontsize=20)

    ax1[5].plot(range(1,len(df_origin)+1),df_origin.iloc[:,5], 'o')
    ax1[5].set_ylabel('value', fontsize=20)
    ax1[5].legend(['c^L'], fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rcParams.update({'font.size': 20})

    fig2, ax2 = plt.subplots(6,1,figsize=(16,16))
    ax2[0].plot(range(1,len(df_processed)+1),df_processed.iloc[:,0], 'o')
    ax2[0].set_ylabel('value', fontsize=20)
    ax2[0].legend(['(K/u)^L'], fontsize=20)

    ax2[1].plot(range(1,len(df_processed)+1),df_processed.iloc[:,1], 'o')
    ax2[1].set_ylabel('value', fontsize=20)
    ax2[1].legend(['(K/u)^M'], fontsize=20)

    ax2[2].plot(range(1,len(df_processed)+1),df_processed.iloc[:,2], 'o')
    ax2[2].set_ylabel('value', fontsize=20)
    ax2[2].legend(['Jm^L'], fontsize=20)

    ax2[3].plot(range(1,len(df_processed)+1),df_processed.iloc[:,3], 'o')
    ax2[3].set_ylabel('value', fontsize=20)
    ax2[3].legend(['Jm^M'], fontsize=20)

    ax2[4].plot(range(1,len(df_processed)+1),df_processed.iloc[:,4], 'o')
    ax2[4].set_ylabel('value', fontsize=20)
    ax2[4].legend(['u^L/u^M'], fontsize=20)

    ax2[5].plot(range(1,len(df_processed)+1),df_processed.iloc[:,5], 'o')
    ax2[5].set_ylabel('value', fontsize=20)
    ax2[5].legend(['c^L'], fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rcParams.update({'font.size': 20})
def results_visualization():
    plt.figure(figsize=(16,8), dpi=80)
    ax1 = plt.subplot(121)
    ax1.set_ylabel('k2', fontsize=20)
    ax1.set_xlabel('ε', fontsize=20)
    plt.tick_params(labelsize=20)
    ax1.scatter(df_origin_macro.iloc[:,6], df_origin_macro.iloc[:,7], s = 48, c = 'g')
    ax1.scatter(df_origin_micro.iloc[:,6], df_origin_micro.iloc[:,7], s = 24, c = 'b')

    ax2 = plt.subplot(122)
    ax2.set_ylabel('k2', fontsize=20)
    ax2.set_xlabel('ε', fontsize=20)
    plt.tick_params(labelsize=20)
    ax2.scatter(df_processed_macro.iloc[:,6], df_processed_macro.iloc[:,7], s = 48, c = 'g')
    ax2.scatter(df_processed_micro.iloc[:,6], df_processed_micro.iloc[:,7], s = 24, c = 'b')

class ANN_L1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(ANN_L1, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden)
        self.predict_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        hidden_result = self.hidden_layer(x)
        relu_result = F.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return predict_result

class ANN_L2(torch.nn.Module):
    # Net class initialization method
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        # succeed initialization method from father class
        super(ANN_L2, self).__init__()
        # build the first hidden layer
        self.hidden_layer1 = torch.nn.Linear(n_feature, n_hidden1)
        # build the second hidden layer
        self.hidden_layer2 = torch.nn.Linear(n_hidden1, n_hidden2)
        # output layer, receive hidden layer signals
        self.predict_layer = torch.nn.Linear(n_hidden2, n_output)

    # forward propagation
    def forward(self, x):
        hidden_result1 = self.hidden_layer1(x)
        relu_result1 = F.relu(hidden_result1)
        hidden_result2 = self.hidden_layer2(relu_result1)
        relu_result2 = F.relu(hidden_result2)
        predict_result = self.predict_layer(relu_result2)
        return predict_result

class ANN_L3(torch.nn.Module):
    # Net class initialization method
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(ANN_L3, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_feature, n_hidden1)# first hidden layer
        self.hidden_layer2 = torch.nn.Linear(n_hidden1, n_hidden2)# second hidden layer
        self.hidden_layer3 = torch.nn.Linear(n_hidden2, n_hidden3)# third hidden layer
        self.predict_layer = torch.nn.Linear(n_hidden3, n_output)# output layer

    # forward propagation
    def forward(self, x):
        hidden_result1 = self.hidden_layer1(x)
        relu_result1 = F.relu(hidden_result1)
        hidden_result2 = self.hidden_layer2(relu_result1)
        relu_result2 = F.relu(hidden_result2)
        hidden_result3 = self.hidden_layer3(relu_result2)
        relu_result3 = F.relu(hidden_result3)
        predict_result = self.predict_layer(relu_result3)
        return predict_result

def ANN_Linear_train_test(n_train, n_hidden_layer, learning_rate, backward = False):

    # divide dataset into train set and test set
    df_train_macro, df_test_macro = train_test_split(df_processed_macro, train_size = 0.7)
    df_train_micro, df_test_micro = train_test_split(df_processed_micro, train_size = 0.7)
    df_train = pd.concat( [df_train_macro, df_train_micro], axis=0)
    df_test = pd.concat( [df_test_macro, df_test_micro], axis=0)
    #reverse the input and output data
    if backward == False:
        #transform into tensor
        train_input_tensor = torch.tensor(df_train.iloc[:,0:6].values)
        train_output_tensor = torch.tensor(df_train.iloc[:,6:].values)
        test_input_tensor = torch.tensor(df_test.iloc[:,0:6].values)
        test_output_tensor = torch.tensor(df_test.iloc[:,6:].values)
        dim_input = 6
        dim_output = 2
    else:
        train_input_tensor = torch.tensor(df_train.iloc[:,6:].values)
        train_output_tensor = torch.tensor(df_train.iloc[:,0:6].values)
        test_input_tensor = torch.tensor(df_test.iloc[:,6:].values)
        test_output_tensor = torch.tensor(df_test.iloc[:,0:6].values)
        dim_input = 2
        dim_output = 6

    #initialization of ANN
    if len(n_hidden_layer) == 1:
        net = ANN_L1(n_feature = dim_input, n_hidden = n_hidden_layer[0], n_output = dim_output)
    if len(n_hidden_layer) == 2:
        net = ANN_L2(n_feature = dim_input, n_hidden1 = n_hidden_layer[0], n_hidden2 = n_hidden_layer[1], n_output = dim_output)
    if len(n_hidden_layer) == 3:
        net = ANN_L3(n_feature = dim_input, n_hidden1 = n_hidden_layer[0], n_hidden2 = n_hidden_layer[1], n_hidden3 = n_hidden_layer[2], n_output = dim_output)
    print(net)

    # train net with optimization method chose
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # error computation method
    loss_func = torch.nn.MSELoss(reduction='sum')

    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.set_ylabel('error', fontsize=20)
    ax1.set_xlabel('epoch', fontsize=20)
    plt.tick_params(labelsize=20)

    #storage set up
    error = pd.DataFrame(columns=['epoch', 'error'])

    # start training and illustrate the
    for i in range(n_train):
        # input data to predict
        prediction = net(train_input_tensor.float())

        # error between expected data and predicted data, pay attention on consequence
        # first is predicted, second is real value
        loss = loss_func(prediction, train_output_tensor.float())

        # start optimization
        # set gradient as 0 before each optimization
        optimizer.zero_grad()

        # error pachpropogation
        loss.backward()

        # optimize parameters based on minimum loss
        optimizer.step()

        if i > 0:
            error = error.append(pd.DataFrame({'epoch':[i], 'error':[loss.data.numpy()]}))

    ax1.scatter(error['epoch'],error['error'])

    #estimation of training
    prediction = net(test_input_tensor.float())
    loss = loss_func(prediction, test_output_tensor.float())
    df_prediction_point = pd.DataFrame(prediction.detach().numpy())

    print('The main square error of trained net in test set (expected and predicted): ' + str(loss.data.numpy()))

    if backward == False:
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.set_ylabel('k2', fontsize=20)
        ax2.set_xlabel('ε', fontsize=20)
        plt.tick_params(labelsize=20)
        ax2.scatter(df_prediction_point.iloc[:,0], df_prediction_point.iloc[:,1], s = 48, c = 'g')
        ax2.scatter(df_test.iloc[:,6], df_test.iloc[:,7], s = 24, c = 'b')
    else:
        fig2, ax2 = plt.subplots(6,1,figsize=(16,16))
        ax2[0].plot(range(1,len(df_test)+1),df_test.iloc[:,0], 'o')
        ax2[0].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,0], 'ro')
        ax2[0].set_ylabel('value', fontsize=20)
        ax2[0].legend(['(K/u)^L Ex', '(K/u)^L Pr'], fontsize=20)

        ax2[1].plot(range(1,len(df_test)+1),df_test.iloc[:,1], 'o')
        ax2[1].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,1], 'ro')
        ax2[1].set_ylabel('value', fontsize=20)
        ax2[1].legend(['(K/u)^M Ex', '(K/u)^M Pr'], fontsize=20)

        ax2[2].plot(range(1,len(df_test)+1),df_test.iloc[:,2], 'o')
        ax2[2].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,2], 'ro')
        ax2[2].set_ylabel('value', fontsize=20)
        ax2[2].legend(['Jm^L Ex', 'Jm^L Pr'], fontsize=20)

        ax2[3].plot(range(1,len(df_test)+1),df_test.iloc[:,3], 'o')
        ax2[3].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,3], 'ro')
        ax2[3].set_ylabel('value', fontsize=20)
        ax2[3].legend(['Jm^M Ex', 'Jm^M Pr'], fontsize=20)

        ax2[4].plot(range(1,len(df_test)+1),df_test.iloc[:,4], 'o')
        ax2[4].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,4], 'ro')
        ax2[4].set_ylabel('value', fontsize=20)
        ax2[4].legend(['u^L/u^M Ex', 'u^L/u^M Pr'], fontsize=20)

        ax2[5].plot(range(1,len(df_test)+1),df_test.iloc[:,5], 'o')
        ax2[5].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,5], 'ro')
        ax2[5].set_ylabel('value', fontsize=20)
        ax2[5].legend(['c^L Ex', 'c^L Pr'], fontsize=20)
        plt.rcParams.update({'font.size': 20})


    return df_prediction_point

def ANN_Linear_train_fig(n_train, n_hidden_layer, learning_rate, backward = False):

    # divide dataset into train set and test set
    df_train_macro, df_test_macro = train_test_split(df_processed_macro, train_size = 0.7)
    df_train_micro, df_test_micro = train_test_split(df_processed_micro, train_size = 0.7)
    df_train = pd.concat( [df_train_macro, df_train_micro], axis=0)
    df_test = pd.concat( [df_test_macro, df_test_micro], axis=0)
    #reverse the input and output data
    if backward == False:
        #transform into tensor
        train_input_tensor = torch.tensor(df_train.iloc[:,0:6].values)
        train_output_tensor = torch.tensor(df_train.iloc[:,6:].values)
        test_input_tensor = torch.tensor(df_test.iloc[:,0:6].values)
        test_output_tensor = torch.tensor(df_test.iloc[:,6:].values)
        dim_input = 6
        dim_output = 2
    else:
        train_input_tensor = torch.tensor(df_train.iloc[:,6:].values)
        train_output_tensor = torch.tensor(df_train.iloc[:,0:6].values)
        test_input_tensor = torch.tensor(df_test.iloc[:,6:].values)
        test_output_tensor = torch.tensor(df_test.iloc[:,0:6].values)
        dim_input = 2
        dim_output = 6

    #initialization of ANN
    if len(n_hidden_layer) == 1:
        net = ANN_L1(n_feature = dim_input, n_hidden = n_hidden_layer[0], n_output = dim_output)
    if len(n_hidden_layer) == 2:
        net = ANN_L2(n_feature = dim_input, n_hidden1 = n_hidden_layer[0], n_hidden2 = n_hidden_layer[1], n_output = dim_output)
    if len(n_hidden_layer) == 3:
        net = ANN_L3(n_feature = dim_input, n_hidden1 = n_hidden_layer[0], n_hidden2 = n_hidden_layer[1], n_hidden3 = n_hidden_layer[2], n_output = dim_output)
    print(net)

    # train net with optimization method chose
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # error computation method
    loss_func = torch.nn.MSELoss(reduction='sum')

    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.set_ylabel('error', fontsize=20)
    ax1.set_xlabel('epoch', fontsize=20)
    plt.tick_params(labelsize=20)

    #storage set up
    error = pd.DataFrame(columns=['epoch', 'error'])

    # start training and illustrate the
    for i in range(n_train):
        # input data to predict
        prediction = net(train_input_tensor.float())

        # error between expected data and predicted data, pay attention on consequence
        # first is predicted, second is real value
        loss = loss_func(prediction, train_output_tensor.float())

        # start optimization
        # set gradient as 0 before each optimization
        optimizer.zero_grad()

        # error pachpropogation
        loss.backward()

        # optimize parameters based on minimum loss
        optimizer.step()

        if i > 0:
            error = error.append(pd.DataFrame({'epoch':[i], 'error':[loss.data.numpy()]}))

    ax1.scatter(error['epoch'],error['error'])

    #estimation of training
    prediction = net(test_input_tensor.float())
    loss = loss_func(prediction, test_output_tensor.float())
    df_prediction_point = pd.DataFrame(prediction.detach().numpy())

    print('The main square error of trained net in test set (expected and predicted): ' + str(loss.data.numpy()))

    if backward == False:
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.set_ylabel('k2', fontsize=20)
        ax2.set_xlabel('ε', fontsize=20)
        plt.tick_params(labelsize=20)
        ax2.scatter(df_prediction_point.iloc[:,0], df_prediction_point.iloc[:,1], s = 48, c = 'g')
        ax2.scatter(df_test.iloc[:,6], df_test.iloc[:,7], s = 24, c = 'b')
    else:
        fig2, ax2 = plt.subplots(6,1,figsize=(16,16))
        ax2[0].plot(range(1,len(df_test)+1),df_test.iloc[:,0], 'o')
        ax2[0].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,0], 'ro')
        ax2[0].set_ylabel('value', fontsize=20)
        ax2[0].legend(['(K/u)^L Ex', '(K/u)^L Pr'], fontsize=20)

        ax2[1].plot(range(1,len(df_test)+1),df_test.iloc[:,1], 'o')
        ax2[1].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,1], 'ro')
        ax2[1].set_ylabel('value', fontsize=20)
        ax2[1].legend(['(K/u)^M Ex', '(K/u)^M Pr'], fontsize=20)

        ax2[2].plot(range(1,len(df_test)+1),df_test.iloc[:,2], 'o')
        ax2[2].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,2], 'ro')
        ax2[2].set_ylabel('value', fontsize=20)
        ax2[2].legend(['Jm^L Ex', 'Jm^L Pr'], fontsize=20)

        ax2[3].plot(range(1,len(df_test)+1),df_test.iloc[:,3], 'o')
        ax2[3].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,3], 'ro')
        ax2[3].set_ylabel('value', fontsize=20)
        ax2[3].legend(['Jm^M Ex', 'Jm^M Pr'], fontsize=20)

        ax2[4].plot(range(1,len(df_test)+1),df_test.iloc[:,4], 'o')
        ax2[4].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,4], 'ro')
        ax2[4].set_ylabel('value', fontsize=20)
        ax2[4].legend(['u^L/u^M Ex', 'u^L/u^M Pr'], fontsize=20)

        ax2[5].plot(range(1,len(df_test)+1),df_test.iloc[:,5], 'o')
        ax2[5].plot(range(1,len(df_prediction_point)+1),df_prediction_point.iloc[:,5], 'ro')
        ax2[5].set_ylabel('value', fontsize=20)
        ax2[5].legend(['c^L Ex', 'c^L Pr'], fontsize=20)
        plt.rcParams.update({'font.size': 20})


    return fig2
