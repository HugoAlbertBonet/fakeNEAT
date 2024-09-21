import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from sklearn import datasets
from torch.nn import functional as F
import random
import copy
import time

def shuffle(a, b, seed):
   rand_state = np.random.RandomState(seed)
   rand_state.shuffle(a)
   rand_state.seed(seed)
   rand_state.shuffle(b)

@dataclass
class Config:

    proportion_train = 0.3
    proportion_val = 0.5
    hidden_layers = 10
    population_size = 100
    num_generations = 10000
    mutation_rate = 0.1
    top_n = int(population_size*mutation_rate)

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

    #training parameters
    learning_rate = 0.01
    train_iters = 5

    #evaluation parameters
    eval_iters = 5
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
               if Config.acc: loss = torch.autograd.Variable(1-acc, requires_grad= True)
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

def evaluate_population(population, suma = 0, max_fitness = 0, min_fitness = np.inf,  split = "val", top_n = False):
    res = []
    for indiv in population:
        fit = evaluate_fitness(indiv, split).item()
        res.append(fit)
        suma += fit
        if fit > max_fitness: max_fitness = fit
        if fit < min_fitness: min_fitness = fit
    if top_n:
        indices = sorted(range(len(res)), key=lambda i: res[i], reverse=True)[:Config.top_n]
        return res, suma, max_fitness, min_fitness, [copy.deepcopy(res[i]) for i in indices], [copy.deepcopy(population[i]) for i in indices]
    return res, suma, max_fitness, min_fitness

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

def mutation_remove_neuron(individual): pass

def mutation_mini_train(indiv):
    individual = copy.deepcopy(indiv)
    optimizer = torch.optim.AdamW(individual.network.parameters(), lr = Config.learning_rate)

    for iter in range(Config.train_iters):
        xb, yb = get_batch("train")
        logits, loss = individual.network(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
    return individual

def crossover(parent1, parent2):
    pass

def best_selection(fitness):
    selected = torch.multinomial(nn.Softmax(0)(torch.FloatTensor([(1 - fit) for fit in fitness])), int(Config.mutation_rate*Config.population_size))
    return sorted(selected.tolist(), reverse= True)

def worst_selection(fitness, suma, min_fitness):
    selected = torch.multinomial(nn.Softmax(0)(torch.FloatTensor([(fit - min_fitness) for fit in fitness])), 3*int(Config.mutation_rate*Config.population_size))
    return sorted(selected.tolist(), reverse= True)

def delete_worst(selected, population, fitness, suma):
    for idx in selected:
        population[idx] = population[-1]
        fitness[idx] = fitness[-1]
        population.pop() 
        fit = fitness.pop()
        suma -= fit

    return suma

def compare_best(new_fit, best_fit, new_pop, best_pop):
    lista = zip(new_fit + best_fit, range(len(new_fit + best_fit)))
    ordered_list = sorted(lista)
    fit, indices = zip(*ordered_list)
    pop = new_pop+best_pop
    return [copy.deepcopy(f) for f in fit[:len(best_fit)]], [copy.deepcopy(pop[i]) for i in indices[:len(best_fit)]]





##############################
#         MAIN LOOP          #
##############################

if __name__ == "__main__":
    population = create_population()
    fitness, suma, max_fitness, min_fitness, best_fitness, best_population = evaluate_population(population, top_n = True)
    for i in range(Config.num_generations): 
        t1 = time.time()
        best_selected = best_selection(fitness)
        t2 = time.time()
        new_individuals = [mutation_add_neuron(population[i]) for i in best_selected]
        t3 = time.time()
        best_selected = best_selection(fitness)
        t4 = time.time()
        new_individuals = new_individuals + [mutation_mini_train(population[i]) for i in best_selected]
        t5 = time.time()
        new_fitness, suma, max_fitness, min_fitness = evaluate_population(new_individuals, suma, max_fitness, min_fitness)
        t6 = time.time()
        best_fitness, best_population = compare_best(new_fitness, best_fitness, new_individuals, best_population )
        t7 = time.time()

        population = population + new_individuals
        fitness = fitness + new_fitness

        t8 = time.time()
        worst_selected = worst_selection(fitness, suma, min_fitness)
        t9 = time.time()
        suma = delete_worst(worst_selected, population, fitness, suma)
        t10 = time.time()
        population = population + best_population
        fitness = fitness + best_fitness
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)

        print(f"End of generation {i+1}, best fitness: {min_fitness:2f}, worst fitness: {max_fitness:2f}, {len(population)}")
        print(f"""Time Analysis:
              - Time spent on First Selection: {t2-t1}
              - Time spent on Adding Neurons: {t3-t2}
              - Time spent on Mini Train: {t5-t4}
              - Time spent on evaluating new population: {t6-t5}
              - Time spent comparing new-best: {t7-t6}
              - Time spent on Worst Selection: {t9-t8}
              - Time spent on deletion: {t10-t9}""")







    