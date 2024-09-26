# fakeNEAT

FakeNEAT is an implementation of genetic algorithms for the architectural optimization and training of neural networks based on NeuroEvolution of Augmented Topologies (NEAT) [1]. FakeNEAT is focused on maintaining the philosophy of NEAT but simplifies and modifies certain aspects of the original implementation to optimize inference time once the optimized neural network is trained.

FakeNEAT, at the moment, is in charge of:
- Creating a population on networks based on your dataset, taking into account if it requires a classification or regression task and the dimension of the features and variables to predict. Now the initial population is diverse, achieving better results, thanks to the possibility to add a random number of neurons to each layer (whose maximum number can be established by the programmer).
- Evolving the population of networks through generations, using a fitness function to evaluate the performance of each. This fitness function can vary among MSE for regression tasks or Cross Entropy Loss or 1 - Accuracy for classification tasks. 
- Encoder-Decoder crossover: Divide parents into two parts; encoder and decoder, and cross encoder1 with decoder2 and viceversa.
- Mutate solutions, selected with a softmax probability distribution based on the fitness of each, in three different ways: , by adding a random number of new neurons to a random layer of the neural network, by removing a random number of new neurons from a random layer of the neural network, and by performing a mini-train with a small percentage of the train set, aiming for fast implementation. 
- Keeping the top N individuals (based on the mutation proportion) to ensure they do not disappear (a bugfix has been implemented here to reduce elitism). 
- Keeping track of the development of the training process in three different ways.

The next planned implementations are:
- Adding the possibility of performing crossover between neural networks following two different methods:
    - Equivalent number of neurons crossover: Add or eliminate neurons from each layer to create a child equivalent to parent1 but with the number of neurons of each layer of parent2, and viceversa.
    - Matrix swap: Locate 2 consecutive layers with the same number of neurons in both networks and swap their weights.
- Optimize comparison of best results with new results as it is the most costly function.
- Perform experiments on new datasets.
- Start with digists with little mutations.
- Mutation with random probability for each individual.
- Try other selection processes.

FakeNEAT has been able to achieve the following results: 
| Dataset        | Metric           | Best result  | Iterations |
| ------------- |:-------------:| :-----:|-----:|
| Iris (Sklearn)      | Accuracy | 0.983 | 1500 |
| Digits (Sklearn)    | Accuracy | 0.5 | 2000 |

## References

[1] K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," in Evolutionary Computation, vol. 10, no. 2, pp. 99-127, June 2002, doi: 10.1162/106365602320169811.
keywords: {Genetic algorithms;neural networks;neuroevolution;network topologies;speciation;competing conventions},


