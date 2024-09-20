import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from sklearn import datasets
from torch.nn import functional as F
import random
import copy

def shuffle(a, b, seed):
   rand_state = np.random.RandomState(seed)
   rand_state.shuffle(a)
   rand_state.seed(seed)
   rand_state.shuffle(b)

@dataclass
class Config:

    proportion_train = 0.2
    proportion_val = 0.6
    hidden_layers = 10
    population_size = 100
    num_generations = 10000
    mutation_rate = 0.05

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
    splits = {"train": (data[:int(proportion_train*len(data))], target[:int(proportion_train*len(target))]),
              "val": (data[int(proportion_train*len(data)): int((proportion_train + proportion_val)*len(data))], target[int(proportion_train*len(target)): int((proportion_train + proportion_val)*len(target))]),
              "test": (data[int((proportion_train + proportion_val)*len(data)):], target[int((proportion_train + proportion_val)*len(target)):])}

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
        self.module_list = nn.ModuleList([LinearModule() for i in range(Config.hidden_layers-1)])
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
def evaluate_fitness(indiv, split, acc = False): 
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
    return [evaluate_fitness(indiv, split) for indiv in population]

def mutation_add_neuron(individual):

    indiv = copy.deepcopy(individual)

    idx = random.randint(0, Config.hidden_layers-1)
    indiv.gen[idx] += 1

    if idx == Config.hidden_layers-1:
        indiv.network.out_layer.in_features = indiv.gen[idx]
        j, i = indiv.network.out_layer.weight.shape
        indiv.network.out_layer.weight = nn.Parameter(torch.cat((indiv.network.out_layer.weight, torch.rand(j, 1)), dim = 1))
    else:
        indiv.network.module_list[idx].linear.in_features = indiv.gen[idx]
        j, i = indiv.network.module_list[idx].linear.weight.shape
        indiv.network.module_list[idx].linear.weight = nn.Parameter(torch.cat((indiv.network.module_list[idx].linear.weight, torch.rand(j, 1)), dim = 1))
    if idx == 0:
        j, i = indiv.network.in_layer.linear.weight.shape
        indiv.network.in_layer.linear.out_features = indiv.gen[idx]
        indiv.network.in_layer.linear.weight = nn.Parameter(torch.cat((indiv.network.in_layer.linear.weight, torch.rand(1, i)), dim = 0))
    else:
        j, i = indiv.network.module_list[idx-1].linear.weight.shape
        indiv.network.module_list[idx-1].linear.out_features = indiv.gen[idx]
        indiv.network.module_list[idx-1].linear.weight = nn.Parameter(torch.cat((indiv.network.module_list[idx-1].linear.weight, torch.rand(1, i)), dim = 0))

    return indiv

def crossover(parent1, parent2):
    pass

def best_selection(fitness):
    suma = np.sum(fitness)
    max_fitness = np.max(fitness)
    min_fitness = np.min(fitness)
    selected = []
    for i in range(int(Config.mutation_rate*Config.population_size)):
        while True:
            idx = random.randint(0, len(fitness)-1)
            if idx not in selected and random.random() < (min_fitness + max_fitness - fitness[idx])/suma:
                selected.append(idx)
                break
    return selected

def worst_selection(fitness):
    suma = np.sum(fitness)
    selected = []
    for i in range(int(Config.mutation_rate*Config.population_size)):
        while True:
            idx = random.randint(0, len(fitness)-1)
            if idx not in selected and random.random() < fitness[idx]/suma:
                selected.append(idx)
                break
    return selected

def delete_worst(selected):
    pass



##############################
#         MAIN LOOP          #
##############################

if __name__ == "__main__":
    population = create_population()
    fitness = evaluate_population(population)

    best_selected = best_selection(fitness)
    new_individuals = [mutation_add_neuron(population[i]) for i in best_selected]
    new_fitness = evaluate_population(new_individuals)

    population = population + new_individuals
    fitness = fitness + new_fitness

    worst_selected = worst_selection(fitness)
    delete_worst(worst_selected)





    