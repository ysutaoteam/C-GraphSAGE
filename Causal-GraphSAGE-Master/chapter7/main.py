import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from chapter7.Net import GraphSage
from chapter7.Data import CoraData
from chapter7.causal_sampling import multihop_sampling
from chapter7.sampling import multihop_sampling1
from collections import namedtuple


INPUT_DIM = 1433
HIDDEN_DIM = [128, 7]   # Number of hidden cell nodes
NUM_NEIGHBORS_LIST = [15, 10]   #  The number of neighbors in each order of sampling
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 20
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20   # Number of batches per epoch cycle
LEARNING_RATE = 0.01     # Learning rate
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE="cpu"
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

data = CoraData().data
x = data.x / data.x.sum(1, keepdims=True)

train_index = np.where(data.train_mask)[0]
train_label = data.y[train_index]
test_index = np.where(data.test_mask)[0]
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  #Adam优化器



# Add noise according to Bernoulli distribution
x_gauss = np.zeros((2708, 1433))
#The second parameter is perturbation, ratio
obs = np.random.binomial(1,0.7,1433)
con_Us = np.random.randint(1, 3, 2708)
obs_x = np.zeros((1, 1433))
# U
for i, val in enumerate(con_Us):
    if con_Us[i] == 1:
        for j, val1 in enumerate(data.x[i]):
            if obs[j] == 1:
                obs_x[0][j] = val1
            elif obs[j] == 0:
                if val1 == 1:
                    obs_x[0][j] = 0
                elif val1 == 0:
                    obs_x[0][j] = 1
        x_gauss[i] = obs_x
    elif con_Us[i] == 2:
        x_gauss[i] = data.x[i]
xc = x_gauss / x_gauss.sum(1, keepdims=True)  # Normalize the data



def train():
    model.train()
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
                batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
                # The training node is selected randomly as the source node
                batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
                # Causal sampling
                batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict, con_Us)
                batch_sampling_x = [torch.from_numpy(xc[idx]).float().to(DEVICE) for idx in batch_sampling_result]
                # Get the feature vector corresponding to the sampled node
                batch_train_logits = model(batch_sampling_x)
                # (BATCH_SIZE,hidden_size[-1]=7)
                loss = criterion(batch_train_logits, batch_src_label)
                # Clear gradient
                optimizer.zero_grad()
                # Gradient of back propagation calculation parameters
                loss.backward()
                # Gradient update using optimization method
                optimizer.step()
                print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()

def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling1(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        # Test data perturbed
        # test_x = [torch.from_numpy(xc[idx]).float().to(DEVICE) for idx in test_sampling_result]
        # Test data non-perturbed
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        # (len(test_index),hidden_size[-1]=7)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        # Label corresponding to test node
        predict_y = test_logits.max(1)[1]
        # The prediction label takes argmax for the prediction result by row
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        # Calculate the accuracy on the test node
        print("Test Accuracy: ", accuarcy)


if __name__ == '__main__':
    train()

