import numpy as np
import matplotlib.pyplot as plt

#  (x, y)
x_data = np.linspace(-10, 10, 100)
# true: y = 2x^3 - 4x^2 + x - 3 + random
y_data = 2 * x_data**3 - 4 * x_data**2 + x_data - 3 + np.random.normal(0, 10, size=x_data.shape)

# parametrler umumi 
population_size = 50
polynomial_degree = 12  

def initialize_population(size, degree):
    # for each individuall -------
    return np.random.randn(size, degree + 1)

population = initialize_population(population_size, polynomial_degree)

def calculate_fitness(individual, x, y):
    # evaluate polynomial
    y_pred = np.polyval(individual, x)
    # calculate mean squared error
    mse = np.mean((y - y_pred) ** 2)
    return -mse  

fitness_scores = np.array([calculate_fitness(ind, x_data, y_data) for ind in population])

def select_parents(population, fitness, num_parents):
    # Select the best individuals based on fitness
    parents_indices = np.argsort(fitness)[-num_parents:]
    return population[parents_indices]

num_parents = 20
parents = select_parents(population, fitness_scores, num_parents)

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # select parents
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

offspring_size = (population_size - num_parents, polynomial_degree + 1)
offspring = crossover(parents, offspring_size)

def mutate(offspring, mutation_rate=0.01):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                #mutation
                offspring[idx, gene_idx] += np.random.randn()
    return offspring

mutated_offspring = mutate(offspring)

num_generations = 1000

for generation in range(num_generations):
    fitness_scores = np.array([calculate_fitness(ind, x_data, y_data) for ind in population])
    parents = select_parents(population, fitness_scores, num_parents)
    offspring = crossover(parents, offspring_size)
    mutated_offspring = mutate(offspring)
    population[0:num_parents, :] = parents
    population[num_parents:, :] = mutated_offspring

    #STEP_BEST
    best_fitness = np.max(fitness_scores)
    print(f"Generation {generation}: Best Fitness = {best_fitness}")

#best solution
best_individual = population[np.argmax(fitness_scores)]
print("Best polynomial coefficients found:", best_individual)

#BEST FIT
y_best_fit = np.polyval(best_individual, x_data)

#neticeler
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_best_fit, color='red', label='Best Fit')
plt.legend()
plt.show()
