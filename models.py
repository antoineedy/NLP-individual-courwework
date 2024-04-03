import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification

from torch.autograd import Variable


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity="relu"
        )

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class Models:
    """
    rnn, pretrained
    """

    def __init__(self, method):
        self.method = method
        self.model = None
        if self.method == "pretrained":
            self.model = AutoModelForTokenClassification.from_pretrained(
                "surrey-nlp/roberta-base-finetuned-abbr"
            )
        elif self.method == "rnn":
            input_dim = 100  # input dimension
            hidden_dim = 100  # hidden layer dimension
            layer_dim = 1  # number of hidden layers
            output_dim = 4  # output dimension
            self.model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
        else:
            raise ValueError("Invalid method")

    def fit(self, train_loader=None, test_loader=None, num_epochs=5, input_dim=100):
        if self.method == "pretrained":
            pass
        elif self.method == "rnn":
            error = nn.CrossEntropyLoss()
            learning_rate = 0.05
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            seq_dim = 50
            loss_list = []
            iteration_list = []
            accuracy_list = []
            count = 0
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):

                    train = Variable(images.view(-1, seq_dim, input_dim))
                    labels = Variable(labels)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward propagation
                    outputs = self.model(train)

                    # Calculate softmax and ross entropy loss
                    print(outputs.shape)
                    print(labels.shape)
                    loss = error(outputs, labels)

                    # Calculating gradients
                    loss.backward()

                    # Update parameters
                    optimizer.step()

                    count += 1

                    if count % 250 == 0:
                        # Calculate Accuracy
                        correct = 0
                        total = 0
                        # Iterate through test dataset
                        for images, labels in test_loader:
                            images = Variable(images.view(-1, seq_dim, input_dim))

                            # Forward propagation
                            outputs = self.model(images)

                            # Get predictions from the maximum value
                            predicted = torch.max(outputs.data, 1)[1]

                            # Total number of labels
                            total += labels.size(0)

                            correct += (predicted == labels).sum()

                        accuracy = 100 * correct / float(total)

                        # store loss and iteration
                        loss_list.append(loss.data)
                        iteration_list.append(count)
                        accuracy_list.append(accuracy)
                        if count % 500 == 0:
                            # Print Loss
                            print(
                                "Iteration: {}  Loss: {}  Accuracy: {} %".format(
                                    count, loss.data[0], accuracy
                                )
                            )
        else:
            raise ValueError("Invalid method")

    def predict(self, X_test):
        if self.method == "rnn":
            seq_dim = 28
            input_dim = 100
            X_test = Variable(X_test.view(-1, seq_dim, input_dim))
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            return predicted
        elif self.method == "pretrained-bert":
            inputs = X_test
            labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(
                0
            )  # Batch size 1
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            out = torch.argmax(logits, dim=2)
            out
            return out
