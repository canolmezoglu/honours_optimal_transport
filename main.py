from math import exp
from random import seed
from random import random



class Network:
    def __init__(self, n_inputs, n_hidden,n_outputs):
        self.network = [[[random() for i in range(n_inputs+1)] for i in range(n_hidden)]]  \
                       + [[[random() for i in range(n_hidden+1)] for i in range(n_outputs)]]
        self.output = [[0 for i in range(n_hidden)]] + [[0 for i in range(n_outputs)]]
        self.delta = [[0 for i in range(n_hidden)]] + [[0 for i in range(n_outputs)]]

    def activate(self,layer,neuron_index,inputs):
        weights = self.network[layer][neuron_index]
        bias = weights[-1]
        activation = 0
        #dont look at the last element as it is bias
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation+bias

    def transfer(self,activation):
        return 1.0 / (1.0 + exp(-activation))

    def forward_propagate(self,row):
        inputs = row
        for layer in range(len(self.network)):
            new_inputs = []
            for neuron in range(len(self.network[layer])):
                activation = self.activate(layer,neuron,inputs)
                self.output[layer][neuron] = self.transfer(activation)
                new_inputs.append(self.output[layer][neuron])
            inputs = new_inputs
        return inputs

    def transfer_derivative(self,i,j):
        output = self.output[i][j]
        return output * (1.0 - output)

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            if i == len(self.network) -1:
                for j in range(len(layer)):
                    self.delta[i][j] = (self.output[i][j] - expected[j]) * self.transfer_derivative(i,j)
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for weights in self.network[i+1]:
                        error += (weights[j] * self.delta[i+1][j])
                    self.delta[i][j] = error * self.transfer_derivative(i,j)


# Update network weights with error
    def update_weights(self,row,l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = self.output[i - 1]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron[j] -= l_rate * self.delta[i][j] * inputs[j]
                neuron[j] -= l_rate * self.delta[i][j]

    # Train a network for a fixed number of epochs
    def train_network(self,train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)


    def predict(self, row):
        outputs = self.forward_propagate( row)
        return outputs.index(max(outputs))

    def __str__(self):
        grr = ""
        for layer in self.network:
            grr += str(layer) + "\n"
        return grr


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(1)
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = Network(n_inputs, 2, n_outputs)
    network.train_network(dataset, 0.8, 1000, n_outputs)
    for row in dataset:
        prediction = network.predict(row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
