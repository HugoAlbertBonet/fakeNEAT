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
import math

def shuffle(a, b, seed):
   rand_state = np.random.RandomState(seed)
   rand_state.shuffle(a)
   rand_state.seed(seed)
   rand_state.shuffle(b)

@dataclass
class ConfigNEGA:
    """
    Data class to set the parameters for:
    - The development of the genetic algorithm
    - The dataset to be used
    - The process of training and evaluation
    """

    verbose = 1
    # verbose = 0 for no text alerts
    # verbose = 1 for just one line that rewrites itself
    # verbose = 2 for just one line of text without rewriting
    # verbose = 3 for adding time information of each step

    proportion_train = 0.5                              # Proportion of dataset used for mini-train
    proportion_val = 0.3                                # Proportion of dataset used to evaluate the individuals during genetic algorithm
    hidden_layers = 5                                   # Number of hidden layers per network
    max_neurons = 500                                   # Max number of neurons per layer in inicialization
    population_size = 100                               # Size of population
    num_generations = 150                              # Max number of iterations of genetic algorithm
    mutation_rate = 0.1                                 # Proportion of individuals of population suffering mutations
    top_n = 3                                           # Number of best results to keep from disappearing
    survivors = int(0.5*population_size)                # In case destruction of pupulation is implemented, how many survivors
    destruction_iters = 500                             # Number of iterations between destructions
    crossovers = 1                                      # Number of pairs of solutions crossed at each generation
    limit_val = 0.06                                    # Minimum error in validation for early stopping
    limit_test = 0.06                                   # If early stopping is triggered, minimum error in test to consider successful

    # Loading the datasets
    name_dataset = "iris"
    data, target = np.float32(datasets.load_iris().data), datasets.load_iris().target
    shuffle(data, target, 42) # Remove if shuffling is not necessary
    data, target = torch.from_numpy(data), torch.from_numpy(target)
    
    classifier = True                   # Is it a classification task?
    onehot = False                      # Are the labels already in OneHotEncoding?
    acc = True                          # Do you want to use inverse accuracy (1 - Accuracy) as the fitness function?
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
                  "val": 64,
                  "test": 64}

@dataclass
class ConfigNESA:
    verbose = 1

    T0 = 0.01
    T_final = 0
    max_iters = 3000
    max_time = 5*60
    min_delta = 0.01
    childs = 50


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
        self.in_layer = LinearModule(in_size= ConfigNEGA.in_size, out_size=gen[0])
        self.module_list = nn.ModuleList([LinearModule(in_size = gen[i], out_size= gen[i+1]) for i in range(ConfigNEGA.hidden_layers-1)])
        self.out_layer = nn.Linear(gen[-1], ConfigNEGA.out_size, bias = False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x, targets = None):
        x = self.in_layer(x)
        for module in self.module_list:
            x = module(x)
        x = self.out_layer(x)
        if ConfigNEGA.classifier: 
            x = self.softmax(x)
        if targets is None:
            loss = None
        else:
            if ConfigNEGA.classifier:
               _, y = targets.max(dim=1)
               _, y_pred = x.max(dim=1)
               acc = torch.sum(y_pred == y)/len(y)
               loss = nn.CrossEntropyLoss()(x, y)
               if ConfigNEGA.acc: loss = torch.autograd.Variable(1-acc, requires_grad= True)
            else:
                loss = F.mse_loss(x, targets)
        return x, loss

class Individual:
    def __init__(self):
        self.gen = [random.randint(1, ConfigNEGA.max_neurons) for i in range(ConfigNEGA.hidden_layers)]
        self.network = Network(self.gen)
        self.fitness = 0

    def __call__(self, x, targets = None):
        return self.network(x, targets)



###############################
#      GENETIC ALGORITHM      #
###############################

class NEGA:

    def __init__(self):
        self.best_evolution = []
        self.best_population = []
        self.best_fitness = []


    def create_population(self, n = ConfigNEGA.population_size):
        """
        Creates a population of neural networks
        """
        return [Individual() for i in range(n)]

    def get_batch(self, split: {"train", "val", "test"}):
        """
        Returns a batch of features and targets from the selected split
        """
        x, y = ConfigNEGA.splits[split]
        perm = torch.randperm(x.size(0))
        idx = perm[:ConfigNEGA.batch_size[split]]
        return x[idx], y[idx]
        
    @torch.no_grad()
    def evaluate_fitness(self, indiv: Individual, split: {"train", "val", "test"}): 
        """
        Evaluates the fitness of a neural network on the selected split
        """
        out = {}
        indiv.network.eval() #sets to evaluation phase, with our model it does nothing
        losses = torch.zeros(ConfigNEGA.eval_iters)
        for k in range(ConfigNEGA.eval_iters):
            x, y = self.get_batch(split)
            _, loss = indiv.network(x, y)
            losses[k] = loss.item()
        out = losses.mean()
        indiv.network.train() #sets to training phase, with our model it does nothing
        return out

    def evaluate_population(self, population: list, suma:float = 0, max_fitness:float = 0, min_fitness:float = np.inf,  split:{"train", "val", "test"} = "val", top_n:bool = False):
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
            fit = self.evaluate_fitness(indiv, split).item()
            res.append(fit)
            suma += fit
            if fit > max_fitness: max_fitness = fit
            if fit < min_fitness: min_fitness = fit
        if top_n:
            indices = sorted(range(len(res)), key=lambda i: res[i], reverse=True)[:ConfigNEGA.top_n]
            best_fitness = [copy.deepcopy(res[i]) for i in indices]
            best_population = [copy.deepcopy(population[i]) for i in indices]
            return res, suma, max_fitness, min_fitness, best_fitness, best_population, indices
        return res, suma, max_fitness, min_fitness

    def mutation_add_neuron(self, individual:Individual, index = None, new = None):
        """
        Adds a random number of neurons to a random hidden layer of the individual and returns the mutated neural network
        """

        indiv = copy.deepcopy(individual)

        if index is None:
            idx = random.randint(0, ConfigNEGA.hidden_layers-1)
        else:
            idx = index
        if new is None:
            new_neurons = random.randint(1, 10)
        else:
            new_neurons = new
        indiv.gen[idx] += new_neurons

        if idx == ConfigNEGA.hidden_layers-1:
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

    def mutation_remove_neuron(self, individual:Individual): 
        """
        Removes a neuron from a random hidden layer without eliminating any layer and returns the mutated neural network
        """
        
        indiv = copy.deepcopy(individual)

        idx = random.randint(0, ConfigNEGA.hidden_layers-1)
        if indiv.gen[idx] > 1:
            new_neurons = random.randint(1, indiv.gen[idx]-1)
            indiv.gen[idx] -= new_neurons
            if idx == ConfigNEGA.hidden_layers-1:
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


    def mutation_mini_train(self, indiv:Individual):
        """
        Performs a mini classical training process on a neural network
        """
        individual = copy.deepcopy(indiv)
        optimizer = torch.optim.AdamW(individual.network.parameters(), lr = ConfigNEGA.learning_rate)

        for iter in range(ConfigNEGA.train_iters):
            xb, yb = self.get_batch("train")
            logits, loss = individual.network(xb, yb)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()
        return individual

    def crossover_encoder_decoder(self, parent1, parent2):
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
                parent2 = self.mutation_add_neuron(parent2, idx, parent1.gen[idx] - parent2.gen[idx])
            elif parent2.gen[idx] > parent1.gen[idx]:
                parent1 = self.mutation_add_neuron(parent1, idx, parent2.gen[idx] - parent1.gen[idx])
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


    def best_selection(self, fitness, n = int(ConfigNEGA.mutation_rate*ConfigNEGA.population_size)):
        selected = torch.multinomial(nn.Softmax(0)(torch.FloatTensor([(1 - fit) for fit in fitness])), n)
        return sorted(selected.tolist(), reverse= True)

    def worst_selection(self, fitness, suma, min_fitness):
        selected = torch.multinomial(nn.Softmax(0)(torch.FloatTensor([(fit - min_fitness) for fit in fitness])), 3*int(ConfigNEGA.mutation_rate*ConfigNEGA.population_size) + 2*ConfigNEGA.crossovers)
        return sorted(selected.tolist(), reverse= True)

    def delete_worst(self, selected, population, fitness, suma):
        for idx in selected:
            population[idx] = population[-1]
            fitness[idx] = fitness[-1]
            population.pop() 
            fit = fitness.pop()
            suma -= fit

        return suma

    def compare_best(self, new_fit, best_fit, new_pop, best_pop):
        lista = zip(new_fit + best_fit, range(len(new_fit + best_fit)))
        ordered_list = sorted(lista)
        fit, indices = zip(*ordered_list)
        pop = new_pop+best_pop
        return list(fit[:len(best_fit)]), [copy.deepcopy(pop[i]) for i in indices[:len(best_fit)]], list(fit[len(best_fit):]), [copy.deepcopy(pop[i]) for i in indices[len(best_fit):]]

    def destruct_population(self, population, fitness):
        survivor_indices = random.sample(range(len(population)), ConfigNEGA.survivors)
        return [population[i] for i in survivor_indices], [fitness[i] for i in survivor_indices]

    def __call__(self, save = True):
        population = self.create_population()
        fitness, suma, max_fitness, min_fitness, self.best_fitness, self.best_population, indices = self.evaluate_population(population, top_n = True)
        self.best_evolution = []
        for i in sorted(indices, reverse=True):
            fitness[i] = fitness[-1]
            population[i] = population[-1]
            fitness.pop()
            population.pop()

        for i in range(ConfigNEGA.num_generations): 

            new_individuals = []
            tcross1 = time.time()
            for _ in range(ConfigNEGA.crossovers):
                parents_indices = self.best_selection(fitness + self.best_fitness, n = 2)
                child1, child2 = self.crossover_encoder_decoder(*((population + self.best_population)[i] for i in parents_indices))
                new_individuals = new_individuals + [child1, child2]
            
            t1 = time.time()
            best_selected = self.best_selection(fitness + self.best_fitness)
            t2 = time.time()
            new_individuals = new_individuals + [self.mutation_add_neuron((population + self.best_population)[i]) for i in best_selected]
            t3 = time.time()
            best_selected = self.best_selection(fitness + self.best_fitness)
            t4 = time.time()
            new_individuals = new_individuals + [self.mutation_remove_neuron((population + self.best_population)[i]) for i in best_selected]
            t5 = time.time()
            best_selected = self.best_selection(fitness + self.best_fitness)
            t6 = time.time()
            new_individuals = new_individuals + [self.mutation_mini_train((population + self.best_population)[i]) for i in best_selected]
            t7 = time.time()
            new_fitness, suma, max_fitness, min_fitness = self.evaluate_population(new_individuals, suma, max_fitness, min_fitness)
            t8 = time.time()
            self.best_fitness, self.best_population, new_fitness, new_individuals = self.compare_best(new_fitness, self.best_fitness, new_individuals, self.best_population )
            t9 = time.time()

            population = population + new_individuals
            fitness = fitness + new_fitness

            t10 = time.time()
            worst_selected = self.worst_selection(fitness, suma, min_fitness)
            t11 = time.time()
            suma = self.delete_worst(worst_selected, population, fitness, suma)
            t12 = time.time()

            max_fitness = np.max(fitness)
            min_fitness = np.min(self.best_fitness)
            self.best_evolution.append(min_fitness)
            min_fit_test = self.evaluate_fitness(self.best_population[0], "test")

            if ConfigNEGA.verbose == 1:
                print(f"End of generation {i+1}, best fitness: {min_fitness:.2f}, worst: {max_fitness:.2f}, best in test: {min_fit_test:.2f}, {len(population)}", end = "\r")
            else:
                if ConfigNEGA.verbose > 1:
                    print(f"End of generation {i+1}, best fitness: {min_fitness:.2f}, worst: {max_fitness:.2f}, best in test: {min_fit_test:.2f}, {len(population)}, {self.best_fitness}")
                if ConfigNEGA.verbose > 2:
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
            if min_fitness < ConfigNEGA.limit_val: 
                if min_fit_test < ConfigNEGA.limit_test: break

        if save:
            df = pd.DataFrame({"best fitness": self.best_evolution, "iteration": range(1, len(self.best_evolution)+1)})
            plt.plot(df["iteration"], df["best fitness"])
            plt.savefig(f"./images/{ConfigNEGA.name_dataset}_hiddenlayers{ConfigNEGA.hidden_layers}_maxneurons{ConfigNEGA.max_neurons}_population_size{ConfigNEGA.population_size}_mutation_rate{ConfigNEGA.mutation_rate}_crossovers{ConfigNEGA.crossovers}_fitness{min_fitness}_test{min_fit_test}.png")

            torch.save(self.best_population[0].network, f"./models/{ConfigNEGA.name_dataset}_{min_fitness}.pt")
            with open("./models/genes.txt", "a") as f:
                f.write(f'{ConfigNEGA.name_dataset}_{min_fitness}.pt' + ": " + str(self.best_population[0].gen))

        return self.best_population, self.best_fitness

    def get_best(self):
        return self.best_population, self.best_fitness
    
    def plot_evolution(self):
        df = pd.DataFrame({"best fitness": self.best_evolution, "iteration": range(1, len(self.best_evolution)+1)})
        plt.plot(df["iteration"], df["best fitness"])

###############################
#     SIMULATED ANNEALING     #
###############################

class NESA:
    def __init__(self):
        self.T0 = ConfigNESA.T0
        self.time_spent = 0
        self.ga = NEGA()
        self.ratio = (ConfigNESA.T0 - ConfigNESA.T_final)/ConfigNESA.max_iters
        self.sol_act = None
        self.fit_act = 1

    def length(self, temperature):
        return ConfigNESA.childs

    def new_solution(self, solution):
        idx = random.randint(0, 2)
        if idx == 0:
            return self.ga.mutation_mini_train(solution), idx
        elif idx == 1:
            return self.ga.mutation_remove_neuron(solution), idx
        else:
            return self.ga.mutation_add_neuron(solution), idx 

    def update(self, temperature, iterations):
        return temperature - self.ratio
    
    def get_best(self):
        return self.sol_act, self.fit_act

    def __call__(self, use_NEGA = True, solucion_inicial = None):
        T_act = self.T0
        iterations = 0
        t0 = time.time()
        t1 = time.time()
        self.time_spent = t1-t0
        if use_NEGA:
            best_population, best_fitness = self.ga(save = False)
            self.sol_act = best_population[0]
            self.fit_act = float(best_fitness[0])
        else:
            assert solucion_inicial is not None
            self.sol_act = solucion_inicial[0]
            self.fit_act = float(solucion_inicial[1])

        while (self.fit_act > ConfigNESA.min_delta
               and iterations < ConfigNESA.max_iters 
               and T_act > ConfigNESA.T_final
               and self.time_spent < ConfigNESA.max_time):
            
            
            for count in range(self.length(T_act)):
                sol_cand, idx = self.new_solution(self.sol_act)
                fit_cand = float(self.ga.evaluate_fitness(sol_cand, "val"))

                delta = fit_cand - self.fit_act
                if (delta < 0 or random.random() < math.e**(-delta/T_act)):
                    self.sol_act = sol_cand
                    self.fit_act = fit_cand
            if ConfigNESA.verbose > 0:
                print(f"NESA: End of iteration {iterations}, Current solution fitness: {self.fit_act:.2f}, {idx}")
            iterations += 1
            T_act = self.update(T_act, iterations=iterations)

        
        print(self.fit_act > ConfigNESA.min_delta)
        print(iterations < ConfigNESA.max_iters) 
        print(T_act > ConfigNESA.T_final)
        print(self.time_spent < ConfigNESA.max_time)
        return self.sol_act, self.fit_act

##############################
#         MAIN LOOP          #
##############################

if __name__ == "__main__":
    sa = NESA()
    print(sa())