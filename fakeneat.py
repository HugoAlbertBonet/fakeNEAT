import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from sklearn import datasets
from torch.nn import functional as F

def shuffle(a, b, seed):
   rand_state = np.random.RandomState(seed)
   rand_state.shuffle(a)
   rand_state.seed(seed)
   rand_state.shuffle(b)

@dataclass
class Config:
    data, target = np.float32(datasets.load_iris().data), datasets.load_iris().target
    shuffle(data, target, 42)
    data, target = torch.from_numpy(data), torch.from_numpy(target)
    classifier = True
    onehot = False
    acc = True
    if classifier and not onehot:
        target = F.one_hot(target.to(torch.int64))
    in_size = data.shape[1]
    if len(target.shape) < 2:
        out_size = 1
    else:
        out_size = target.shape[1]
    hidden_layers = 10
    population_size = 100
    splits = {"train": (data[:int(0.3*len(data))], target[:int(0.3*len(target))]),
              "val": (data[int(0.3*len(data)): int(0.6*len(data))], target[int(0.3*len(target)): int(0.6*len(target))]),
              "test": (data[int(0.6*len(data)):], target[int(0.6*len(target)):])}

    #evaluation parameters
    eval_iters = 100
    batch_size = {"train": 32,
                  "val": 32,
                  "test": 32}



##############################
#   NEURAL NETWORK MODULES   #
##############################

class LinearModule(nn.Module):
    def __init__(self, in_size= 1):
        super().__init__()
        self.linear = nn.Linear(in_size, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = LinearModule(Config.in_size)
        self.module_list = nn.ModuleList([LinearModule() for i in range(Config.hidden_layers)])
        self.out_layer = nn.Linear(1, Config.out_size, bias = False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x, targets = None):
        x = self.in_layer(x)
        for module in self.module_list:
            x = module(x)
        x = self.out_layer(x)
        if Config.classifier: 
            x = self.softmax(x)
        if targets is None:
            loss = None
        else:
            if Config.classifier:
               _, y = targets.max(dim=1)
               _, y_pred = x.max(dim=1)
               acc = torch.sum(y_pred == y)/len(y)
               loss = nn.CrossEntropyLoss()(x, y)
               if Config.acc: loss = 1-acc
            else:
                loss = F.mse_loss(x, targets)
        return x, loss

class Individual:
    def __init__(self):
        self.network = Network()
        self.gen = [1 for i in range(Config.hidden_layers)]
        self.fitness = 0

    def __call__(self, x, targets = None):
        return self.network(x, targets)

###############################
# GENETIC ALGORITHM FUNCTIONS #
###############################


def create_population():
    return [Individual() for i in range(Config.population_size)]

def get_batch(split):
    x, y = Config.splits[split]
    perm = torch.randperm(x.size(0))
    idx = perm[:Config.batch_size[split]]
    return x[idx], y[idx]
    

@torch.no_grad()
def fitness(indiv, split, acc = False): 
    out = {}
    indiv.network.eval() #sets to evaluation phase, with our model it does nothing
    losses = torch.zeros(Config.eval_iters)
    for k in range(Config.eval_iters):
        x, y = get_batch(split)
        _, loss = indiv.network(x, y)
        losses[k] = loss.item()
    out = losses.mean()
    indiv.network.train() #sets to training phase, with our model it does nothing
    return out

def evaluate_population(population, split = "val"):
    return [fitness(indiv, split) for indiv in population]



##############################
#         MAIN LOOP          #
##############################

if __name__ == "__main__":
    individual = Individual()
    iris = datasets.load_iris()
    #print(individual.network.module_list[0].linear.weight)
    iris.data= np.float32(iris.data)
    #print(iris.data.shape, Config.data.shape)
    x, y = get_batch("train")
    print(x.shape, y.shape)
    #print(individual.network)
    print(fitness(individual, "train"))

    