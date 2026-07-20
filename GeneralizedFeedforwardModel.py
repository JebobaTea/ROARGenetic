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

        if weights is None:
            self.weights = []
            for n in range(1, len(self.topology)):
                self.weights.append(np.random.randn(self.topology[n], self.topology[n - 1]))
        if biases is None:
            self.biases = []
            for n in range(1, len(self.topology)):
                self.biases.append(np.random.randn(self.topology[n], 1))
        if activation_types is None:
            self.activation_types = []
            for n in range(1, len(self.topology) - 1):
                self.activation_types.append(random.choice(self.allowed_activation_types))
            self.activation_types.append(sigmoid)

    # terrible practice and terrible solution, but we ball
    def generate_name(self):
        # probably a more elegant way to do this but wc
        topology_str_flattened = ""
        for k in self.topology:
            topology_str_flattened += str(k)
        return self._lineage + "_" + topology_str_flattened

    def debug_structure(self):
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

    def insert_hidden_layer(self, idx, neurons):
        # Assumes operation not performed with first or last index
        self.topology.insert(idx, neurons)
        self.weights.insert(idx, np.ones((self.topology[idx + 1], neurons)))
        self.biases.insert(idx, np.zeros((neurons, 1)))
        self.activation_types.insert(idx, relu)

    # brocken will fix later
    def add_node(self, layer):
        # Assumes operation not performed on first or last layers
        self.topology[layer] += 1

        # Absolutely horrid fix due to first layer being dummies
        idx = layer - 1
        self.biases[idx] = np.vstack((self.biases[idx], [0]))

        new_weights_current = np.ones((1, self.topology[layer - 1]))
        self.weights[idx] = np.vstack((self.weights[idx], new_weights_current))

        new_weights_following = np.random.randn(self.topology[layer + 1], 1)
        self.weights[idx + 1] = np.hstack((self.weights[idx + 1], new_weights_following))

    def feed_forward(self, A0):
        inp = A0.T

        for i in range(len(self.weights)):
            out_raw = self.weights[i] @ inp + self.biases[i]
            activation = self.activation_types[i]
            out_final = activation(out_raw)
            inp = out_final

        return out_final

    def mutate_activation_functions(self, prob):
        for i in range(0, len(self.topology) - 1):
            if random.random() < prob:
                self.topology[i] = random.choice(self.allowed_activation_types)

    def mutate_node_insertion(self, prob):
        if random.random() < prob:
            self.add_node(random.randint(1, len(self.topology) - 2))

    def mutate_layer_insertion(self, prob):
        if random.random() < prob:
            idx = random.randint(1, len(self.topology) - 2)
            self.insert_hidden_layer(idx, self.topology[idx])

    def mutate_nodes(self, prob, mode="replace"):
        for i in range(len(self.weights)):
            if mode == "nudge":
                self.weights[i], self.biases[i] = nudge_random_node_parameters(self.weights[i], self.biases[i], prob)
            elif mode == "replace":
                self.weights[i], self.biases[i] = replace_random_node_parameters(self.weights[i], self.biases[i], prob)
            else:
                print("No mutation performed; invalid mutation type")

    def nudge_random_parameters(self, prob):
        pass

    def replace_random_parameters(self, prob):
        pass

    # TODO further reading https://link.springer.com/article/10.1007/s10710-024-09481-7
    # TODO neat https://macwha.medium.com/evolving-ais-using-a-neat-algorithm-2d154c623828

    def check_topology_match(self, other):
        if self.topology == other.topology:
            return True
        return False

    def dirty_speciated_blockswap_crossover(self, other):
        pass

    def dirty_speciated_average_crossover(self, other):
        pass

    def dirty_blockswap_crossover(self, other):
        pass

    def dirty_average_crossover(self, other):
        pass

    # aliasing
    is_same_species = check_topology_match
    dirty_specswap = dirty_speciated_blockswap_crossover
    dirty_specavg = dirty_speciated_average_crossover
    dirty_blockswap = dirty_blockswap_crossover
    dirty_avg = dirty_average_crossover

def sigmoid(matrix):
    return 1 / (1 + np.exp(-1 * matrix))

def tanh(matrix):
    return np.tanh(matrix)

def relu(matrix):
    return np.maximum(0, matrix)

def leaky_relu(matrix, alpha=0.01):
    return np.where(matrix > 0, matrix, matrix * alpha)

def nudge_random_node_parameters(weights, biases, chance):
    for index, node in enumerate(weights):
        if (random.random() < chance):
            for weight in node:
                weight += random.uniform(-1, 1)
            biases[index] += random.uniform(-1, 1)
    return weights, biases

def replace_random_node_parameters(weights, biases, chance):
    for index, node in enumerate(weights):
        if (random.random() < chance):
            for w in node:
                w = np.random.randn()
            biases[index] = np.random.randn()
    return weights, biases

def clone(base, uid=1):
    return GeneralizedFeedforwardModel(base.topology, base.weights, base.biases, base.activation_types, base.generate_name(), uid)

def save(target, path):
    with open(path, "wb") as f:
        np.savez(f, topology=np.array(target.topology, dtype=object), weights=np.array(target.weights, dtype=object), biases=np.array(target.biases, dtype=object), activation_types=export_activation(target.activation_types))

def load(path):
    with open(path, "rb") as f:
        file = np.load(f, allow_pickle=True)
        return GeneralizedFeedforwardModel(file['topology'].tolist(), file['weights'].tolist(), file['biases'].tolist(), import_activation(file['activation_types']))

mref = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu
}

def export_activation(arr):
    res = []
    for i in arr:
        res.append([k for k,v in mref.items() if v == i][0])
    return res

def import_activation(arr):
    res = []
    for i in arr:
        res.append(mref[i])
    return res

def test():
    model = GeneralizedFeedforwardModel()
    inp = [[0.3, 0.5, 0.1, 0.9, 0.6]]
    inp = numpy.array(inp)
    model.debug_structure()
    print(model.feed_forward(inp))
    model.mutate_layer_insertion(1)
    model.mutate_node_insertion(1)
    model.debug_structure()
    print(model.feed_forward(inp))
    save(model, "modelsavetest_0.npz")
    model2 = load("modelsavetest_0.npz")
    print(model2.feed_forward(inp))
