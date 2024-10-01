import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from sklearn import datasets
from torch.nn import functional as F
import random
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd

def shuffle(a, b, seed):
   rand_state = np.random.RandomState(seed)
   rand_state.shuffle(a)
   rand_state.seed(seed)
   rand_state.shuffle(b)

@dataclass
class Config:
    """
    Data class to set the parameters for:
    - The development of the genetic algorithm
    - The dataset to be used
    - The process of training and evaluation
    """

    verbose = 2

    proportion_train = 0.5
    proportion_val = 0.4
    hidden_layers = 3
    max_neurons = 150
    population_size = 100
    num_generations = 5000
    mutation_rate = 0.1
    top_n = 5 #int(population_size*mutation_rate)
    survivors = int(0.5*population_size)
    destruction_iters = 500
    crossovers = 10

    data, target = np.float32(datasets.load_digits().data), datasets.load_digits().target
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
    train_iters = 1

    #evaluation parameters
    eval_iters = 1
    batch_size = {"train": 32,
                  "val": 32,
                  "test": 32}





##############################
#   NEURAL NETWORK MODULES   #
##############################

class LinearModule(nn.Module):
    def __init__(self, in_size= 1, out_size = 1):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

class Network(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.in_layer = LinearModule(in_size= Config.in_size, out_size=gen[0])
        self.module_list = nn.ModuleList([LinearModule(in_size = gen[i], out_size= gen[i+1]) for i in range(Config.hidden_layers-1)])
        self.out_layer = nn.Linear(gen[-1], Config.out_size, bias = False)
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
        self.gen = [random.randint(1, Config.max_neurons) for i in range(Config.hidden_layers)]
        self.network = Network(self.gen)
        self.fitness = 0

    def __call__(self, x, targets = None):
        return self.network(x, targets)

###############################
# GENETIC ALGORITHM FUNCTIONS #
###############################


def create_population(n = Config.population_size):
    """
    Creates a population of neural networks
    """
    return [Individual() for i in range(n)]

def get_batch(split: {"train", "val", "test"}):
    """
    Returns a batch of features and targets from the selected split
    """
    x, y = Config.splits[split]
    perm = torch.randperm(x.size(0))
    idx = perm[:Config.batch_size[split]]
    return x[idx], y[idx]
    

@torch.no_grad()
def evaluate_fitness(indiv: Individual, split: {"train", "val", "test"}): 
    """
    Evaluates the fitness of a neural network on the selected split
    """
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

def evaluate_population(population: list, suma:float = 0, max_fitness:float = 0, min_fitness:float = np.inf,  split:{"train", "val", "test"} = "val", top_n:bool = False):
    """
    Evaluates the fitness of each individual from the population and returns_
    - The fitness of each individual
    - The total fitness of the poopulation
    - The maximum fitness of the population
    - The minimum fitness of the population
    - If necessary, the top N individuals and their fitness
    """
    res = []
    for indiv in population:
        fit = evaluate_fitness(indiv, split).item()
        res.append(fit)
        suma += fit
        if fit > max_fitness: max_fitness = fit
        if fit < min_fitness: min_fitness = fit
    if top_n:
        indices = sorted(range(len(res)), key=lambda i: res[i], reverse=True)[:Config.top_n]
        best_fitness = [copy.deepcopy(res[i]) for i in indices]
        best_population = [copy.deepcopy(population[i]) for i in indices]
        return res, suma, max_fitness, min_fitness, best_fitness, best_population, indices
    return res, suma, max_fitness, min_fitness

def mutation_add_neuron(individual:Individual, index = None, new = None):
    """
    Adds a random number of neurons to a random hidden layer of the individual and returns the mutated neural network
    """

    indiv = copy.deepcopy(individual)

    if index is None:
        idx = random.randint(0, Config.hidden_layers-1)
    else:
        idx = index
    if new is None:
        new_neurons = random.randint(1, 10)
    else:
        new_neurons = new
    indiv.gen[idx] += new_neurons

    if idx == Config.hidden_layers-1:
        indiv.network.out_layer.in_features = indiv.gen[idx]
        j, i = indiv.network.out_layer.weight.shape
        indiv.network.out_layer.weight = nn.Parameter(torch.cat((indiv.network.out_layer.weight, torch.rand(j, new_neurons)), dim = 1))
    else:
        indiv.network.module_list[idx].linear.in_features = indiv.gen[idx]
        j, i = indiv.network.module_list[idx].linear.weight.shape
        indiv.network.module_list[idx].linear.weight = nn.Parameter(torch.cat((indiv.network.module_list[idx].linear.weight, torch.rand(j, new_neurons)), dim = 1))
    if idx == 0:
        j, i = indiv.network.in_layer.linear.weight.shape
        indiv.network.in_layer.linear.out_features = indiv.gen[idx]
        indiv.network.in_layer.linear.weight = nn.Parameter(torch.cat((indiv.network.in_layer.linear.weight, torch.rand(new_neurons, i)), dim = 0))
    else:
        j, i = indiv.network.module_list[idx-1].linear.weight.shape
        indiv.network.module_list[idx-1].linear.out_features = indiv.gen[idx]
        indiv.network.module_list[idx-1].linear.weight = nn.Parameter(torch.cat((indiv.network.module_list[idx-1].linear.weight, torch.rand(new_neurons, i)), dim = 0))

    return indiv

def mutation_remove_neuron(individual:Individual): 
    """
    Removes a neuron from a random hidden layer without eliminating any layer and returns the mutated neural network
    """
     
    indiv = copy.deepcopy(individual)

    idx = random.randint(0, Config.hidden_layers-1)
    if indiv.gen[idx] > 1:
        new_neurons = random.randint(1, indiv.gen[idx]-1)
        indiv.gen[idx] -= new_neurons
        if idx == Config.hidden_layers-1:
            indiv.network.out_layer.in_features = indiv.gen[idx]
            j, i = indiv.network.out_layer.weight.shape
            indiv.network.out_layer.weight = nn.Parameter(indiv.network.out_layer.weight[:, :i-new_neurons])
        else:
            indiv.network.module_list[idx].linear.in_features = indiv.gen[idx]
            j, i = indiv.network.module_list[idx].linear.weight.shape
            indiv.network.module_list[idx].linear.weight = nn.Parameter(indiv.network.module_list[idx].linear.weight[:, :i-new_neurons])
        if idx == 0: 
            j, i = indiv.network.in_layer.linear.weight.shape
            indiv.network.in_layer.linear.out_features = indiv.gen[idx]
            indiv.network.in_layer.linear.weight = nn.Parameter(indiv.network.in_layer.linear.weight[:j-new_neurons, :])

        else:
            j, i = indiv.network.module_list[idx-1].linear.weight.shape
            indiv.network.module_list[idx-1].linear.out_features = indiv.gen[idx]
            indiv.network.module_list[idx-1].linear.weight = nn.Parameter(indiv.network.module_list[idx-1].linear.weight[:j-new_neurons, :])

    return indiv


def mutation_mini_train(indiv:Individual):
    """
    Performs a mini classical training process on a neural network
    """
    individual = copy.deepcopy(indiv)
    optimizer = torch.optim.AdamW(individual.network.parameters(), lr = Config.learning_rate)

    for iter in range(Config.train_iters):
        xb, yb = get_batch("train")
        logits, loss = individual.network(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
    return individual

def crossover_encoder_decoder(parent1, parent2):
    """
    Combines the first part of a parent (encoder) with the second part of the other parent (decoder), and viceversa
    """
    parent1, parent2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    child1, child2 = Individual(), Individual()
    indices = []
    for i in range(len(parent1.gen)):
        if parent1.gen[i] == parent2.gen[i]:
            indices.append(i)
    if len(indices) >0:
        idx = indices[random.randint(0, len(indices)-1)]
        child1.gen = parent1.gen[:idx] + parent2.gen[idx:]
        child2.gen = parent2.gen[:idx] + parent1.gen[idx:]

        child1.network.in_layer = parent1.network.in_layer
        child2.network.in_layer = parent2.network.in_layer
        if idx > 0:
            child1.network.module_list = parent1.network.module_list[:idx] + parent2.network.module_list[idx:]
            child2.network.module_list = parent2.network.module_list[:idx] + parent1.network.module_list[idx:]
        else: 
            child1.network.module_list = parent2.network.module_list
            child2.network.module_list = parent1.network.module_list
        child1.network.out_layer = parent2.network.out_layer
        child2.network.out_layer = parent1.network.out_layer
    else:
        idx = random.randint(0, len(parent1.gen)-1)
        if parent1.gen[idx] > parent2.gen[idx]:
            parent2 = mutation_add_neuron(parent2, idx, parent1.gen[idx] - parent2.gen[idx])
        elif parent2.gen[idx] > parent1.gen[idx]:
            parent1 = mutation_add_neuron(parent1, idx, parent2.gen[idx] - parent1.gen[idx])
        child1.gen = parent1.gen[:idx] + parent2.gen[idx:]
        child2.gen = parent2.gen[:idx] + parent1.gen[idx:]

        child1.network.in_layer = parent1.network.in_layer
        child2.network.in_layer = parent2.network.in_layer
        if idx > 0:
            child1.network.module_list = parent1.network.module_list[:idx] + parent2.network.module_list[idx:]
            child2.network.module_list = parent2.network.module_list[:idx] + parent1.network.module_list[idx:]
        else: 
            child1.network.module_list = parent2.network.module_list
            child2.network.module_list = parent1.network.module_list
        child1.network.out_layer = parent2.network.out_layer
        child2.network.out_layer = parent1.network.out_layer



    return child1, child2


def best_selection(fitness, n = int(Config.mutation_rate*Config.population_size), i = 1):
    selected = torch.multinomial(nn.Softmax(0)(torch.FloatTensor([(1 - fit) for fit in fitness])), n)
    return sorted(selected.tolist(), reverse= True)

def worst_selection1(fitness, suma, min_fitness):
    selected = torch.multinomial(nn.Softmax(0)(torch.FloatTensor([(fit - min_fitness) for fit in fitness])), 3*int(Config.mutation_rate*Config.population_size) + 2*Config.crossovers)
    return sorted(selected.tolist(), reverse= True)

def worst_selection2(fitness, suma, min_fitness):
    avg = np.mean(fitness)
    selected = [i for i in range(len(fitness)) if fitness[i] <= avg][:Config.population_size//3]
    return sorted(selected, reverse= True)

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
    return list(fit[:len(best_fit)]), [copy.deepcopy(pop[i]) for i in indices[:len(best_fit)]], list(fit[len(best_fit):]), [copy.deepcopy(pop[i]) for i in indices[len(best_fit):]]

def destruct_population(population, fitness):
    survivor_indices = random.sample(range(len(population)), Config.survivors)
    return [population[i] for i in survivor_indices], [fitness[i] for i in survivor_indices]





##############################
#         MAIN LOOP          #
##############################

if __name__ == "__main__":
    population = create_population()
    fitness, suma, max_fitness, min_fitness, best_fitness, best_population, indices = evaluate_population(population, top_n = True)
    best = []
    worst_selected = []
    for i in sorted(indices, reverse=True):
        fitness[i] = fitness[-1]
        population[i] = population[-1]
        fitness.pop()
        population.pop()

    for i in range(Config.num_generations): 
        new = max(5, (Config.population_size - len(worst_selected))//3)
        new_individuals = []
        tcross1 = time.time()
        for _ in range(Config.crossovers):
            parents_indices = best_selection(fitness + best_fitness, n = 2, i = i)
            child1, child2 = crossover_encoder_decoder(*((population + best_population)[i] for i in parents_indices))
            new_individuals = new_individuals + [child1, child2]
        
        t1 = time.time()
        best_selected = best_selection(fitness + best_fitness, n = new, i = i)
        t2 = time.time()
        new_individuals = new_individuals + [mutation_add_neuron((population + best_population)[i]) for i in best_selected]
        t3 = time.time()
        best_selected = best_selection(fitness + best_fitness, n = new, i = i)
        t4 = time.time()
        new_individuals = new_individuals + [mutation_remove_neuron((population + best_population)[i]) for i in best_selected]
        t5 = time.time()
        best_selected = best_selection(fitness + best_fitness, n = new, i = i)
        t6 = time.time()
        new_individuals = new_individuals + [mutation_mini_train((population + best_population)[i]) for i in best_selected]
        t7 = time.time()
        new_fitness, suma, max_fitness, min_fitness = evaluate_population(new_individuals, suma, max_fitness, min_fitness)
        t8 = time.time()
        best_fitness, best_population, new_fitness, new_individuals = compare_best(new_fitness, best_fitness, new_individuals, best_population )
        t9 = time.time()

        population = population + new_individuals
        fitness = fitness + new_fitness

        t10 = time.time()
        worst_selected = worst_selection1(fitness, suma, min_fitness)
        t11 = time.time()
        suma = delete_worst(worst_selected, population, fitness, suma)
        t12 = time.time()

        worst_selected = worst_selection2(fitness, suma, min_fitness)
        suma = delete_worst(worst_selected, population, fitness, suma)
        population, fitness = population[:Config.population_size], fitness[:Config.population_size]

        max_fitness = np.max(fitness)
        min_fitness = np.min(best_fitness)
        avg_fitness = np.mean(fitness + best_fitness)
        best.append(min_fitness)

        if Config.verbose == 1:
            print(f"End of generation {i+1}, best fitness: {min_fitness:.2f}, worst: {max_fitness:.2f}, mean: {avg_fitness:.2f}, {len(population)}", end = "\r")
        else:
            if Config.verbose > 1:
                print(f"End of generation {i+1}, best fitness: {min_fitness:.2f}, worst: {max_fitness:.2f}, mean: {avg_fitness:.2f}, {len(population)}, {best_fitness}")
            if Config.verbose > 2:
                print(f"""Time Analysis:
              - Time spent on Crossover: {t1-tcross1:.4f}
              - Time spent on First Selection: {t2-t1:.4f}
              - Time spent on Adding Neurons: {t3-t2:.4f}
              - Time spent on Removing Neurons: {t5-t4:.4f}
              - Time spent on Mini-Training: {t7-t6:.4f}
              - Time spent on evaluating new population: {t8-t7:.4f}
              - Time spent comparing new-best: {t9-t8:.4f}
              - Time spent on Worst Selection: {t11-t10:.4f}
              - Time spent on deletion: {t12-t11:.4f}""")

    df = pd.DataFrame({"best fitness": best, "iteration": range(1, Config.num_generations+1)})
    plt.plot(df["iteration"], df["best fitness"])
    plt.savefig(f"./images/digits_hiddenlayers{Config.hidden_layers}_maxneurons{Config.max_neurons}_population_size{Config.population_size}_mutation_rate{Config.mutation_rate}_crossovers{Config.crossovers}_fitness{min_fitness}.png")

    torch.save(best_population[0].network)
    with open("models/genes.txt", "a") as f:
        f.write(f'digits_{min_fitness}.pt' + ": " + best_population[0].gen)
