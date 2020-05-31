import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import copy
import numpy as np
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import random


class Parser:
    def __init__(self):
        self.epochs = 1
        self.lr = 0.001
        self.test_batch_size = 8
        self.batch_size = 8
        self.log_interval = 10
        self.seed = 1
        # self.seq_len = 100
        self.past_history = 1000
        self.future_target = 47
        self.STEP = 100
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 10000
        self.EVALUATION_INTERVAL = 200


args = Parser()
torch.manual_seed(args.seed)
past_history=args.future_target
future_target = args.future_target
STEP = args.future_target
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# def multivariate_data(dataset, target, start_index, end_index, history_size,
  #                     target_size, step, single_step=False):
  # data = []
  # labels = []
  # start_index = start_index + history_size
  # if end_index is None:
  #   end_index = len(dataset) - target_size
  #
  # for i in range(start_index, end_index):
  #   indices = range(i-history_size, i, step)
  #   data.append(dataset[indices])
  #
  #   if single_step:
  #     labels.append(target[i+target_size])
  #   else:
  #     labels.append(target[i:i+target_size])
  #
  # return np.array(data), np.array(labels)


# def transform_data(a,b, seq_len):
#     x, y = [], []
#     for i in range(len(a) - seq_len):
#         x_i = a[i : i + seq_len]
#         x.append(x_i)
#     for i in range(len(b) - seq_len):
#         y_i = b[i : i + seq_len]
#         y.append(y_i)
#     x_arr = np.array(x).reshape(-1, seq_len)
#     y_arr = np.array(y).reshape(-1, seq_len)
#     x_var = Variable(torch.from_numpy(x_arr).float())
#     y_var = Variable(torch.from_numpy(y_arr).float())
#     return x_var, y_var


def load_data(cluster):
    X = pd.read_csv('./dataset/' + cluster + "/X.csv")
    X.rename({'f3':'missing_f3'},axis=1,inplace=True)
    # X.drop(['f4'],axis=1,inplace=True)
    Y = pd.read_csv('./dataset/' + cluster + "/Y.csv")
    df = pd.concat([X,Y],axis=1)
    df.timestamp=df.timestamp*(10**11)
    df = df.sort_values('timestamp')
    # df.set_index('timestamp', inplace=True)
    # df = df['f3']
    dataset = df.values
    TRAIN_SPLIT = int(len(X) * 2 / 3)
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    dataset_train= dataset[:TRAIN_SPLIT]
    dataset_test = dataset[TRAIN_SPLIT:]
    VAL_SPLIT = int(len(dataset_train) * 0.8)
    dataset_val = dataset_train[VAL_SPLIT:]
    dataset_train = dataset_train[:VAL_SPLIT]
    x_train, y_train = split_sequences(dataset_train, args.STEP)
    x_test, y_test = split_sequences(dataset_test, args.STEP)
    x_val, y_val = split_sequences(dataset_val, args.STEP)
    # x_train = np.array(x_train).reshape(-1, args.STEP)
    # y_train = np.array(y_train).reshape(-1, args.STEP)
    # x_test = np.array(x_test).reshape(-1, args.STEP)
    # y_test = np.array(y_test).reshape(-1, args.STEP)
    # x_val = np.array(x_val).reshape(-1, args.STEP)
    # y_val = np.array(y_val).reshape(-1, args.STEP)


    # x_train, y_train = multivariate_data(dataset[:, :-1], dataset[:, -1:], 0,
    #                                                    TRAIN_SPLIT, past_history,
    #                                                    future_target, STEP,
    #                                                    single_step=True)
    # x_test, y_test = multivariate_data(dataset[:,:-1], dataset[:, -1:],
    #                                                TRAIN_SPLIT, None, past_history,
    #                                                future_target, STEP,
    #                                                single_step=True)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)


    # x_train, y_train = transform_data(x_train,y_train, seq_len)
    # x_val, y_val = transform_data(x_val,y_val, seq_len)
    # x_test, y_test = transform_data(x_test,y_test, seq_len)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    train = TensorDataset(x_train, y_train)
    test = TensorDataset(x_test, y_test)
    val = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader, val_loader

# class MV_LSTM(torch.nn.Module):
#     def __init__(self,n_features,seq_length):
#         super(MV_LSTM, self).__init__()
#         self.n_features = n_features
#         self.seq_len = seq_length
#         self.n_hidden = 20 # number of hidden states
#         self.n_layers = 1 # number of LSTM layers (stacked)
#
#         self.l_lstm = torch.nn.LSTM(input_size = n_features,
#                                  hidden_size = self.n_hidden,
#                                  num_layers = self.n_layers,
#                                  batch_first = True)
#         # according to pytorch docs LSTM output is
#         # (batch_size,seq_len, num_directions * hidden_size)
#         # when considering batch_first = True
#         self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
#
#
#     def init_hidden(self, batch_size):
#         # even with batch_first = True this remains same as docs
#         hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
#         cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
#         self.hidden = (hidden_state, cell_state)
#
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         if(not(hasattr(self,'hidden'))):
#             self.init_hidden(batch_size)
#         lstm_out, self.hidden = self.l_lstm(x,self.hidden)
#         # lstm_out(with batch_first = True) is
#         # (batch_size,seq_len,num_directions * hidden_size)
#         # for following linear layer we want to keep batch_size dimension and merge rest
#         # .contiguous() -> solves tensor compatibility error
#         x = lstm_out.contiguous().view(batch_size,-1)
#         return self.l_linear(x)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, future=0, y=None):
        outputs = []

        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=100,
        n_epochs=15,
        do_teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= batch
            self.train_losses.append(train_loss)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            print(
                "Epoch %d Train loss: %.2f. Validation loss: %.2f. Avg future: %.2f. Elapsed time: %.2fs."
                % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed)
            )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= batch
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")


def generate_sequence(scaler, model, x_sample, future=1000):
    """ Generate future values for x_sample with the model """
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred

def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})


# def inverse_transform(scalar, df, columns):
#     for col in columns:
#         df[col] = scaler.inverse_transform(df[col])
#     return df

bob_cl = '7'
alice_cl = '0'
target_cl = '15'
cluster=bob_cl
hook = sy.TorchHook(torch)
# bob = sy.VirtualWorker(hook, id="bob")
# alice = sy.VirtualWorker(hook, id="alice")
kwargs_websocket = {"host": "localhost", "hook": hook}
alice = WebsocketClientWorker(id='alice', port=8779, **kwargs_websocket)
bob = WebsocketClientWorker(id='bob', port=8778, **kwargs_websocket)
compute_nodes = [bob, alice]
bob_train, bob_test, bob_val = load_data(bob_cl)
alice_train, alice_test, alice_val = load_data(alice_cl)
target_train, target_test, target_val = load_data(target_cl)
remote_dataset = (list(), list())
train_distributed_dataset = []

for batch_idx, (data,target) in enumerate(bob_train):
    data = data.send(compute_nodes[0])
    target = target.send(compute_nodes[0])
    remote_dataset[0].append((data, target))

for batch_idx, (data,target) in enumerate(alice_train):
    data = data.send(compute_nodes[1])
    target = target.send(compute_nodes[1])
    remote_dataset[1].append((data, target))

# model_bob = MV_LSTM(n_features=7, seq_length=args.STEP)
model_bob = Model(input_size=7, hidden_size=21, output_size=1)
loss_fn_bob = nn.MSELoss()
optimizer_bob = optim.Adam(model_bob.parameters(), lr=args.lr)
scheduler_bob = optim.lr_scheduler.StepLR(optimizer_bob, step_size=5, gamma=0.1)
optimization_bob = Optimization(model_bob, loss_fn_bob, optimizer_bob, scheduler_bob)

# model_alice = MV_LSTM(n_features=7, seq_length=
model_alice = Model(input_size=7, hidden_size=21, output_size=1)
loss_fn_alice = nn.MSELoss()
optimizer_alice = optim.Adam(model_alice.parameters(), lr=args.lr)
scheduler_alice = optim.lr_scheduler.StepLR(optimizer_alice, step_size=5, gamma=0.1)
optimization_alice = Optimization(model_alice, loss_fn_alice, optimizer_alice, scheduler_alice)

models = [model_bob, model_alice]
optimizers = [optimization_bob, optimization_alice]

optimization_bob.train(bob_train.dataset.tensors[0], bob_train.dataset.tensors[1],
                       bob_val.dataset.tensors[0], bob_val.dataset.tensors[1], do_teacher_forcing=False)
optimization_alice.train(alice_train.dataset.tensors[0], alice_train.dataset.tensors[1],
                       alice_val.dataset.tensors[0], alice_val.dataset.tensors[1], do_teacher_forcing=False)





























#
# def update(data, target, model, optimizer):
#     model.send(data.location)
#     optimizer.zero_grad()
#     prediction = model(data)
#     loss = F.mse_loss(prediction.view(-1), target.reshape(torch.Size([target.shape[0]])))
#     loss.backward()
#     optimizer.step()
#     return model
#
# def train():
#     for data_index in range(len(remote_dataset[0])-1):
#         for remote_index in range(len(compute_nodes)):
#             data, target = remote_dataset[remote_index][data_index]
#             models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])
#         for model in models:
#             model.get()
#         return utils.federated_avg({
#             "bob": models[0],
#             "alice": models[1]
#         })
#
#
# def test(federated_model):
#     federated_model.eval()
#     test_loss = 0
#     total_p = []
#     total_t = []
#     for data, target in target_test:
#         output = federated_model(data)
#         test_loss += F.mse_loss(output.view(-1), target.reshape(torch.Size([target.shape[0]])), reduction='sum').item()
#         predection = output.data.max(1, keepdim=True)[1]
#         p=torch.cat(list(output), 0)
#         t=torch.cat(list(target), 0)
#         total_p.extend(p.tolist())
#         total_t.extend(t.tolist())
#
#     # total_p = torch.cat(total_p, 0)
#     # total_t = torch.cat(total_t, 0)
#     test_loss /= len(target_test.dataset)
#     test_r2 = r2_score(total_t,total_p)
#     print('Test set: Average loss: {:.4f}'.format(test_loss))
#     print('r2= ' + str(test_r2) + "  reversed:  " + str(r2_score(total_p,total_t)))
#     return total_p,total_t

# for epoch in range(args.epochs):
#     start_time = time.time()
#     print(f"Epoch Number {epoch + 1}")
#     federated_model = train()
#     model = federated_model
#     test(federated_model)
#     total_time = time.time() - start_time
#     print('Communication time over the network', round(total_time, 2), 's\n')