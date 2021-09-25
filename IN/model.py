import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# constants
rel_dim = 1

eff_dim = 100
hidden_obj_dim = 100
hidden_rel_dim = 100
last_dim =n_objects=obj_dim = 100
n_relations = n_objects * (n_objects - 1)
n_half = 500


class RelationModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RelationModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size),
			nn.ReLU()
		)

	def forward(self, x):
		'''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
		return self.model(x)


class ObjectModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(ObjectModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, x):
		'''
        Args:
            x: [n_objects, input_size]
        Returns:
            [n_objects, output_size]

        Note: output_size = number of states we want to predict
        '''
		return self.model(x)


class InteractionNetwork(nn.Module):
	def __init__(self, dim_obj, dim_rel, dim_eff, dim_hidden_obj, dim_hidden_rel, dim_x=0):
		super(InteractionNetwork, self).__init__()
		self.rm = RelationModel(dim_obj * 2 + dim_rel, dim_hidden_rel, dim_eff)
		self.om = ObjectModel(dim_obj + dim_eff + dim_x, dim_hidden_obj, last_dim)  # x, y

	def m(self, obj, rr, rs, ra):
		"""
		The marshalling function;
		computes the matrix products ORr and ORs and concatenates them with Ra

		:param obj: object states
		:param rr: receiver relations
		:param rs: sender relations
		:param ra: relation info
		:return:
		"""
		orr = obj.t().mm(rr)   # (obj_dim, n_relations)
		ors = obj.t().mm(rs)   # (obj_dim, n_relations)
		return torch.cat([orr, ors, ra.t()])   # (obj_dim*2+rel_dim, n_relations)

	def forward(self, obj, rr, rs, ra, x=None):
		"""
		objects, sender_relations, receiver_relations, relation_info
		:param obj: (n_objects, obj_dim)
		:param rr: (n_objects, n_relations)
		:param rs: (n_objects, n_relations)
		:param ra: (n_relations, rel_dim)
		:param x: external forces, default to None
		:return:
		"""
		# marshalling function
		b = self.m(obj, rr, rs, ra)   # shape of b = (obj_dim*2+rel_dim, n_relations)

		# relation module
		e = self.rm(b.t())   # shape of e = (n_relations, eff_dim)
		e = e.t()   # shape of e = (eff_dim, n_relations)

		# effect aggregator
		if x is None:
			a = torch.cat([obj.t(), e.mm(rr.t())])   # shape of a = (obj_dim+eff_dim, n_objects)
		else:
			a = torch.cat([obj.t(), x, e.mm(rr.t())])   # shape of a = (obj_dim+ext_dim+eff_dim, n_objects)

		# object module
		p = self.om(a.t())   # shape of p = (n_objects, 2)

		return p


def format_data(data, idx):
    objs = data[idx, :, :]   # (n_objects, obj_dim)
    receiver_r = np.zeros((n_objects, n_relations), dtype=float)
    sender_r = np.zeros((n_objects, n_relations), dtype=float)
    count = 0   # used as idx of relations
    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                receiver_r[i, count] = 1.0
                sender_r[j, count] = 1.0
                count += 1
    r_info = np.zeros((n_relations, rel_dim))
    target = data[idx + n_half, :, :]  # only want vx and vy predictions
    objs = Variable(torch.FloatTensor(objs.float()))
    sender_r = Variable(torch.FloatTensor(sender_r))
    receiver_r = Variable(torch.FloatTensor(receiver_r))
    r_info = Variable(torch.FloatTensor(r_info))
    target = Variable(torch.FloatTensor(target.float()))
    return objs, sender_r, receiver_r, r_info, target


# set up network
interaction_network = InteractionNetwork(obj_dim, rel_dim, eff_dim, hidden_obj_dim, hidden_rel_dim)
optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()

# training
n_epoch = 100

losses = []

# Input MolOpt data
input_adj = np.load('edge_feat_x.npy', allow_pickle=True)
input_edge = np.load('edge_feat_y.npy', allow_pickle=True)

# Sample a subset of data
id_target = np.random.choice(range(len(input_adj)), 1000)
input_adj = input_adj[id_target]
input_edge = input_edge[id_target]

input_adj = [torch.from_numpy(input_adj[i]) for i in range(len(input_adj))]
input_edge = [torch.from_numpy(input_edge[i]) for i in range(len(input_edge))]

npad = 100
input_adj_tmp = torch.zeros(1000, npad, npad)
input_edge_tmp = torch.zeros(1000, npad, npad)
for i in range(len(input_adj)):
    print("Processing", i, "-th graph")
    if input_adj[i].shape[0] == 0:
        input_adj_tmp[i, :, :] = torch.zeros(npad, npad)
    else:
        input_adj_tmp[i, :, :] = F.pad(input_adj[i], (0, npad - input_adj[i].shape[1], 0, npad - input_adj[i].shape[0]), 'constant', 0)
    if input_edge[i].shape[0] == 0:
        input_edge_tmp[i, :, :] = torch.zeros(npad, npad)
    else:
        input_edge_tmp[i, :, :] = F.pad(input_edge[i], (0, npad - input_edge[i].shape[1], 0, npad - input_edge[i].shape[0]), 'constant', 0)

input_adj = input_adj_tmp
input_edge = input_edge_tmp

input_adj = input_adj.numpy()
input_edge = input_edge.numpy()

data_train = torch.cat((torch.from_numpy(input_adj[:n_half].astype("float64")), torch.from_numpy(input_edge[:n_half].astype("float64"))), dim = 0)
data_test = torch.cat((torch.from_numpy(input_adj[n_half:].astype("float64")), torch.from_numpy(input_edge[n_half:].astype("float64"))), dim = 0)

for epoch in range(100):
    print("="*20, "epoch", epoch, "="*20)
    best_loss = np.inf
    for i in range(int(len(data_train)/2)):
        objects, sender_relations, receiver_relations, relation_info, target = format_data(data_train, i)
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        loss = criterion(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(np.sqrt(loss.data))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
    print("best loss:", best_loss)
    
pred = []
tgt = []

for i in range(int(len(data_test)/2)):
    objects, sender_relations, receiver_relations, relation_info, target = format_data(data_train, i)
    predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
    
    tgt.append(target.detach().cpu().numpy())
    pred.append(predicted.detach().cpu().numpy())
    
########################## Evaluation
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from scipy.spatial import distance
def bhattacharyya(h1, h2):
    '''Calculates the Byattacharyya distance of two histograms.'''

    def normalize(h):
        return h / np.sum(h)

    return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))

# En-dist
js = []
# C-dist
bh = []
# wl-sim
ws = []

for i in range(len(tgt)):
    js.append(distance.jensenshannon(tgt[i].reshape(-1).detach().numpy(), pred[i].reshape(-1).detach().numpy()))
    bh.append(bhattacharyya(tgt[i].reshape(-1).detach().numpy(), pred[i].reshape(-1).detach().numpy()))
    ws.append(stats.wasserstein_distance(tgt[i].reshape(-1).detach().numpy(), pred[i].reshape(-1).detach().numpy()))
    
sum(js)/len(js)
sum(bh)/len(bh)
sum(ws)/len(ws)
