from typing import *
from random import choices, randint, randrange, random
from dataclasses import dataclass
from functools import partial

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


@dataclass
class Activity:
    description: str
    importance: int
    hours: float


def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, activities: [Activity], hours_limit: int) -> int:
    if len(genome) != len(activities):
        raise ValueError("The genome and the activities must have the same length")
    hours, total_importance = 0, 0
    for index, activity in enumerate(activities):
        if genome[index] == 1:
            hours += activity.hours
            total_importance += activity.importance

        if hours > hours_limit:  # Invalid solution !
            return 0
    return total_importance


def select_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("The two genomes must be in the same length")
    length = len(a)
    if length < 2:
        return a, b
    random_index = randint(1, length - 1)
    return a[:random_index] + b[random_index:], b[:random_index] + a[random_index:]


def mutation(genome: Genome, number_of_mutations: int = 1, probability: float = 0.5):
    for i in range(number_of_mutations):
        random_index = randrange(len(genome))
        genome[random_index] = genome[random_index] if random() > probability else genome[random_index] ^ 1
    return genome

def run_evolution(
        populate_func: PopulateFunc = generate_population,
        fitness_func: FitnessFunc = fitness,
        selection_func: SelectionFunc = select_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        fitness_limit: int = 50,
        generation_limit: int = 50) -> Tuple[Population, int]:
    population = populate_func()
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )
        if fitness_func(population[0]) >= fitness_limit:
            return population, i

        next_generation = population[:2]
        for j in range(len(population) // 2 - 1):
            parents = selection_func(population, fitness_func)
            descendant_a, descendant_b = crossover_func(parents[0], parents[1])
            descendant_a, descendant_b = mutation_func(descendant_a), mutation_func(descendant_b)
            next_generation.extend([descendant_a, descendant_b])
        population = next_generation

    return population, generation_limit

def genome_to_activities(genome:Genome, activities:[Activity]):
    return [activity.description for digit, activity in zip(genome, activities) if digit == 1]


def main():
    activities = [
        Activity("Udemy", 8, 1.5),
        Activity("Python", 8, 2),
        Activity("Spanish", 10, 0.5),
        Activity("Working out", 6, 0.5),
        Activity("School Assignments", 4, 2),
        Activity("Netflix", 3, 1.5),
        Activity("Walking The Dog", 9, 0.75),
        Activity("Chess", 6, 0.5),
        Activity("Youtube", 2, 1),
        Activity("Friends", 7, 1.5)
    ]
    population, num_of_generations = run_evolution(
        populate_func=partial(
            generate_population, size=10, genome_length=len(activities)
        ),
        fitness_func=partial(
            fitness, activities=activities, hours_limit=6
        ),
        fitness_limit=45,
        generation_limit=100
    )
    print(f"Finished in {num_of_generations} generations")
    for genome in population:
        print(genome_to_activities(genome, activities))

if __name__ == '__main__':
    main()
