import numpy as np
import random
from typing import List, Tuple, Callable, Any
from abc import ABC, abstractmethod

class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    def __init__(self, genes: np.ndarray):
        self.genes = genes
        self.fitness = None
        
    def __str__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness})"

class GeneticAlgorithm:
    """Genetic Algorithm implementation for optimizing time series forecasting parameters."""
    
    def __init__(self, 
                 gene_bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 tournament_size: int = 3):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            gene_bounds: List of (min, max) bounds for each gene
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of best individuals to keep unchanged
            tournament_size: Size of tournament for selection
        """
        self.gene_bounds = gene_bounds
        self.num_genes = len(gene_bounds)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        
    def initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = []
        for _ in range(self.population_size):
            genes = np.array([
                random.uniform(bounds[0], bounds[1]) 
                for bounds in self.gene_bounds
            ])
            individual = Individual(genes)
            self.population.append(individual)
    
    def evaluate_population(self, fitness_function: Callable):
        """Evaluate fitness for all individuals in the population."""
        for individual in self.population:
            individual.fitness = fitness_function(individual.genes)
    
    def tournament_selection(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        return winner
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform uniform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1_genes = np.copy(parent1.genes)
        child2_genes = np.copy(parent2.genes)
        
        # Uniform crossover
        for i in range(self.num_genes):
            if random.random() < 0.5:
                child1_genes[i], child2_genes[i] = child2_genes[i], child1_genes[i]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual's genes."""
        mutated_genes = np.copy(individual.genes)
        
        for i in range(self.num_genes):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1 * (self.gene_bounds[i][1] - self.gene_bounds[i][0])
                mutated_genes[i] += np.random.normal(0, mutation_strength)
                
                # Ensure bounds are respected
                mutated_genes[i] = np.clip(mutated_genes[i], 
                                         self.gene_bounds[i][0], 
                                         self.gene_bounds[i][1])
        
        return Individual(mutated_genes)
    
    def evolve_generation(self, fitness_function: Callable):
        """Evolve the population by one generation."""
        # Evaluate fitness
        self.evaluate_population(fitness_function)
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best individual and fitness history
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = Individual(np.copy(self.population[0].genes))
            self.best_individual.fitness = self.population[0].fitness
        
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.fitness_history.append({
            'best': self.population[0].fitness,
            'average': avg_fitness,
            'worst': self.population[-1].fitness
        })
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = int(self.elitism_rate * self.population_size)
        new_population.extend(self.population[:elite_count])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Ensure exact population size
        self.population = new_population[:self.population_size]
    
    def run(self, fitness_function: Callable, generations: int, verbose: bool = True):
        """
        Run the genetic algorithm for a specified number of generations.
        
        Args:
            fitness_function: Function to evaluate individual fitness
            generations: Number of generations to evolve
            verbose: Whether to print progress
        
        Returns:
            Best individual found
        """
        self.initialize_population()
        
        for generation in range(generations):
            self.evolve_generation(fitness_function)
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}: "
                      f"Best fitness = {self.best_individual.fitness:.6f}, "
                      f"Avg fitness = {self.fitness_history[-1]['average']:.6f}")
        
        return self.best_individual