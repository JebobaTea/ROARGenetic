import numpy
import numpy as np
import random
import asyncio

# TODO: overseer, evaluate current waypoint idx and current time (if possible) to prevent collisions and kill the slower vehicle; kill on collision, stoppage, stuck vehicles

class GeneralizedFeedforwardModel:
    # Initialize with random values. Last layer always uses sigmoid; expected output for our use case is clipped between 0,1 anyways
    def __init__(self, topology=[5, 8, 8, 3], weights=None, biases=None, activation_types=None, parent_string="ARC", uid=0):
        self.weights = weights
        self.biases = biases
        self.topology = topology
        self.activation_types = activation_types
        self.allowed_activation_types = [sigmoid, tanh, relu, leaky_relu]
        self.uid = uid
        # what's an fstring
        self._lineage = parent_string + ">>" + str(uid)

        if not weights:
            self.weights = []
            for n in range(1, len(self.topology)):
                self.weights.append(np.random.randn(self.topology[n], self.topology[n - 1]))
        if not biases:
            self.biases = []
            for n in range(1, len(self.topology)):
                self.biases.append(np.random.randn(self.topology[n], 1))
        if not activation_types:
            self.activation_types = []
            for n in range(1, len(self.topology) - 1):
                self.activation_types.append(random.choice(self.allowed_activation_types))
            self.activation_types.append(sigmoid)

    # terrible practice and terrible solution, but we ball
    async def generate_name(self):
        # probably a more elegant way to do this but wc
        topology_str_flattened = ""
        for k in self.topology:
            topology_str_flattened += str(k)
        return self._identifier + "_" + topology_str_flattened

    async def debug_structure(self):
        print()
        print(self.topology)
        print(self.activation_types)
        print("Biases: ")
        for k in self.biases:
            print(k.shape)
        print("Weights: ")
        for k in self.weights:
            print(k.shape)
        print()

    async def insert_hidden_layer(self, idx, neurons):
        # Assumes operation not performed with first or last index
        self.topology.insert(idx, neurons)
        self.weights.insert(idx, np.ones((self.topology[idx + 1], neurons)))
        self.biases.insert(idx, np.zeros((neurons, 1)))
        self.activation_types.insert(idx, relu)

    # brocken will fix later
    async def add_node(self, layer):
        # Assumes operation not performed on first or last layers
        self.topology[layer] += 1

        # Absolutely horrid fix due to first layer being dummies
        idx = layer - 1
        self.biases[idx] = np.vstack((self.biases[idx], [0]))

        new_weights_current = np.ones((1, self.topology[layer - 1]))
        self.weights[idx] = np.vstack((self.weights[idx], new_weights_current))

        new_weights_following = np.random.randn(self.topology[layer + 1], 1)
        self.weights[idx + 1] = np.hstack((self.weights[idx + 1], new_weights_following))

    async def feed_forward(self, A0):
        inp = A0.T

        for i in range(len(self.weights)):
            out_raw = self.weights[i] @ inp + self.biases[i]
            activation = self.activation_types[i]
            out_final = await activation(out_raw)
            inp = out_final

        return out_final

    async def mutate_activation_functions(self, prob):
        for i in range(0, len(self.topology) - 1):
            if random.random() < prob:
                self.topology[i] = random.choice(self.allowed_activation_types)

    async def mutate_node_insertion(self, prob):
        if random.random() < prob:
            await self.add_node(random.randint(1, len(self.topology) - 2))

    async def mutate_layer_insertion(self, prob):
        if random.random() < prob:
            idx = random.randint(1, len(self.topology) - 2)
            await self.insert_hidden_layer(idx, self.topology[idx])

    async def mutate_nodes(self, prob, mode="replace"):
        for i in range(len(self.weights)):
            if mode == "nudge":
                self.weights[i], self.biases[i] = await nudge_random_node_parameters(self.weights[i], self.biases[i], prob)
            elif mode == "replace":
                self.weights[i], self.biases[i] = await replace_random_node_parameters(self.weights[i], self.biases[i], prob)
            else:
                print("No mutation performed; invalid mutation type")

    async def nudge_random_parameters(self, prob):
        pass

    async def replace_random_parameters(self, prob):
        pass

    # TODO further reading https://link.springer.com/article/10.1007/s10710-024-09481-7
    # TODO neat https://macwha.medium.com/evolving-ais-using-a-neat-algorithm-2d154c623828

    async def check_topology_match(self, other):
        if self.topology == other.topology:
            return True
        return False

    async def dirty_speciated_blockswap_crossover(self, other):
        pass

    async def dirty_speciated_average_crossover(self, other):
        pass

    async def dirty_blockswap_crossover(self, other):
        pass

    async def dirty_average_crossover(self, other):
        pass

    # aliasing
    is_same_species = check_topology_match
    dirty_specswap = dirty_speciated_blockswap_crossover
    dirty_specavg = dirty_speciated_average_crossover
    dirty_blockswap = dirty_blockswap_crossover
    dirty_avg = dirty_average_crossover

async def sigmoid(matrix):
    return 1 / (1 + np.exp(-1 * matrix))

async def tanh(matrix):
    return np.tanh(matrix)

async def relu(matrix):
    return np.maximum(0, matrix)

async def leaky_relu(matrix, alpha=0.01):
    return np.where(matrix > 0, matrix, matrix * alpha)

async def nudge_random_node_parameters(weights, biases, chance):
    for index, node in enumerate(weights):
        if (random.random() < chance):
            for weight in node:
                weight += random.uniform(-1, 1)
            biases[index] += random.uniform(-1, 1)
    return weights, biases

async def replace_random_node_parameters(weights, biases, chance):
    for index, node in enumerate(weights):
        if (random.random() < chance):
            for w in node:
                w = np.random.randn()
            biases[index] = np.random.randn()
    return weights, biases

async def test():
    model = GeneralizedFeedforwardModel()
    inp = [[0.3, 0.5, 0.1, 0.9, 0.6]]
    inp = numpy.array(inp)
    await model.debug_structure()
    print(await model.feed_forward(inp))
    await model.mutate_layer_insertion(1)
    await model.mutate_node_insertion(1)
    await model.debug_structure()
    print(await model.feed_forward(inp))

asyncio.run(test())
