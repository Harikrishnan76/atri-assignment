"""
neural_network.py
-----------------
Feedforward Neural Network from scratch using NumPy.
Supports: Xavier/Random init, ReLU/Sigmoid/Tanh activations,
          SGD/Momentum/Nesterov/RMSProp/Adam/Nadam optimizers,
          Cross-Entropy and Squared Error losses.
"""

import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        input_size=784,
        hidden_layers=[128, 64],
        output_size=10,
        activation='relu',
        weight_init='xavier',
        loss='cross_entropy'
    ):
        """
        Args:
            input_size    : number of input features (784 for Fashion-MNIST)
            hidden_layers : list of ints, neurons per hidden layer e.g. [128, 64]
            output_size   : number of output classes (10)
            activation    : 'relu' | 'sigmoid' | 'tanh'
            weight_init   : 'xavier' | 'random'
            loss          : 'cross_entropy' | 'squared_error'
        """
        self.layers         = [input_size] + hidden_layers + [output_size]
        self.activation_name = activation
        self.weight_init    = weight_init
        self.loss_name      = loss
        self.params         = self._initialize_weights()
        self.cache          = {}
        self.t              = 0          # timestep for Adam/Nadam bias correction
        self.velocity       = {}         # momentum / adam first moment
        self.rms            = {}         # rmsprop / adam second moment

    # ------------------------------------------------------------------ #
    #  Weight initialisation                                               #
    # ------------------------------------------------------------------ #
    def _initialize_weights(self):
        params = {}
        for i in range(len(self.layers) - 1):
            fan_in  = self.layers[i]
            fan_out = self.layers[i + 1]
            if self.weight_init == 'xavier':
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            else:                          # random
                scale = 0.01
            params[f'W{i+1}'] = np.random.randn(fan_in, fan_out) * scale
            params[f'b{i+1}'] = np.zeros((1, fan_out))
        return params

    # ------------------------------------------------------------------ #
    #  Activation functions                                                #
    # ------------------------------------------------------------------ #
    def _activate(self, Z):
        if   self.activation_name == 'relu':    return np.maximum(0, Z)
        elif self.activation_name == 'sigmoid': return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation_name == 'tanh':    return np.tanh(Z)
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def _activate_grad(self, Z):
        if self.activation_name == 'relu':
            return (Z > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(Z) ** 2
        raise ValueError(f"Unknown activation: {self.activation_name}")

    # ------------------------------------------------------------------ #
    #  Softmax                                                             #
    # ------------------------------------------------------------------ #
    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #
    def forward(self, X):
        self.cache = {'A0': X}
        A = X
        L = len(self.layers) - 1
        for i in range(L):
            Z = A @ self.params[f'W{i+1}'] + self.params[f'b{i+1}']
            self.cache[f'Z{i+1}'] = Z
            if i == L - 1:
                A = self._softmax(Z)       # output layer → softmax
            else:
                A = self._activate(Z)      # hidden layer → chosen activation
            self.cache[f'A{i+1}'] = A
        return A

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #
    def compute_loss(self, y_pred, y_true_onehot):
        m = y_true_onehot.shape[0]
        if self.loss_name == 'cross_entropy':
            y_pred = np.clip(y_pred, 1e-10, 1.0)
            return -np.sum(y_true_onehot * np.log(y_pred)) / m
        elif self.loss_name == 'squared_error':
            return np.sum((y_pred - y_true_onehot) ** 2) / (2 * m)
        raise ValueError(f"Unknown loss: {self.loss_name}")

    # ------------------------------------------------------------------ #
    #  Backward pass                                                       #
    # ------------------------------------------------------------------ #
    def backward(self, y_true_onehot):
        m   = y_true_onehot.shape[0]
        grads = {}
        L   = len(self.layers) - 1
        A_L = self.cache[f'A{L}']

        # Output layer gradient depends on loss choice
        if self.loss_name == 'cross_entropy':
            dZ = A_L - y_true_onehot                      # softmax + CE simplifies nicely
        else:  # squared_error
            dZ = (A_L - y_true_onehot) * A_L * (1 - A_L) # chain through softmax

        grads[f'dW{L}'] = (self.cache[f'A{L-1}'].T @ dZ) / m
        grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(L - 1, 0, -1):
            dA = dZ @ self.params[f'W{i+1}'].T
            dZ = dA * self._activate_grad(self.cache[f'Z{i}'])
            grads[f'dW{i}'] = (self.cache[f'A{i-1}'].T @ dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    # ------------------------------------------------------------------ #
    #  Parameter update — all 6 optimizers                                #
    # ------------------------------------------------------------------ #
    def update_params(self, grads, lr=0.001, optimizer='sgd',
                      beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.t += 1
        L = len(self.layers) - 1

        for i in range(1, L + 1):
            dW = grads[f'dW{i}'].copy()
            db = grads[f'db{i}'].copy()

            # L2 weight decay (applied only to weights, not biases)
            if weight_decay > 0:
                dW += weight_decay * self.params[f'W{i}']

            # ---- SGD ------------------------------------------------- #
            if optimizer == 'sgd':
                self.params[f'W{i}'] -= lr * dW
                self.params[f'b{i}'] -= lr * db

            # ---- Momentum -------------------------------------------- #
            elif optimizer == 'momentum':
                if f'vW{i}' not in self.velocity:
                    self.velocity[f'vW{i}'] = np.zeros_like(dW)
                    self.velocity[f'vb{i}'] = np.zeros_like(db)
                self.velocity[f'vW{i}'] = beta1 * self.velocity[f'vW{i}'] + (1 - beta1) * dW
                self.velocity[f'vb{i}'] = beta1 * self.velocity[f'vb{i}'] + (1 - beta1) * db
                self.params[f'W{i}'] -= lr * self.velocity[f'vW{i}']
                self.params[f'b{i}'] -= lr * self.velocity[f'vb{i}']

            # ---- Nesterov Accelerated Gradient ----------------------- #
            elif optimizer == 'nesterov':
                if f'vW{i}' not in self.velocity:
                    self.velocity[f'vW{i}'] = np.zeros_like(dW)
                    self.velocity[f'vb{i}'] = np.zeros_like(db)
                v_prev_W = self.velocity[f'vW{i}'].copy()
                v_prev_b = self.velocity[f'vb{i}'].copy()
                self.velocity[f'vW{i}'] = beta1 * v_prev_W + (1 - beta1) * dW
                self.velocity[f'vb{i}'] = beta1 * v_prev_b + (1 - beta1) * db
                # Nesterov: use next-step lookahead
                self.params[f'W{i}'] -= lr * (beta1 * self.velocity[f'vW{i}'] + (1 - beta1) * dW)
                self.params[f'b{i}'] -= lr * (beta1 * self.velocity[f'vb{i}'] + (1 - beta1) * db)

            # ---- RMSProp --------------------------------------------- #
            elif optimizer == 'rmsprop':
                if f'vW{i}' not in self.rms:
                    self.rms[f'vW{i}'] = np.zeros_like(dW)
                    self.rms[f'vb{i}'] = np.zeros_like(db)
                self.rms[f'vW{i}'] = beta2 * self.rms[f'vW{i}'] + (1 - beta2) * (dW ** 2)
                self.rms[f'vb{i}'] = beta2 * self.rms[f'vb{i}'] + (1 - beta2) * (db ** 2)
                self.params[f'W{i}'] -= lr * dW / (np.sqrt(self.rms[f'vW{i}']) + epsilon)
                self.params[f'b{i}'] -= lr * db / (np.sqrt(self.rms[f'vb{i}']) + epsilon)

            # ---- Adam ------------------------------------------------ #
            elif optimizer == 'adam':
                if f'mW{i}' not in self.velocity:
                    self.velocity[f'mW{i}'] = np.zeros_like(dW)
                    self.velocity[f'mb{i}'] = np.zeros_like(db)
                    self.rms[f'vW{i}']      = np.zeros_like(dW)
                    self.rms[f'vb{i}']      = np.zeros_like(db)
                self.velocity[f'mW{i}'] = beta1 * self.velocity[f'mW{i}'] + (1 - beta1) * dW
                self.velocity[f'mb{i}'] = beta1 * self.velocity[f'mb{i}'] + (1 - beta1) * db
                self.rms[f'vW{i}']      = beta2 * self.rms[f'vW{i}']      + (1 - beta2) * (dW ** 2)
                self.rms[f'vb{i}']      = beta2 * self.rms[f'vb{i}']      + (1 - beta2) * (db ** 2)
                mW_hat = self.velocity[f'mW{i}'] / (1 - beta1 ** self.t)
                mb_hat = self.velocity[f'mb{i}'] / (1 - beta1 ** self.t)
                vW_hat = self.rms[f'vW{i}']      / (1 - beta2 ** self.t)
                vb_hat = self.rms[f'vb{i}']      / (1 - beta2 ** self.t)
                self.params[f'W{i}'] -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
                self.params[f'b{i}'] -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)

            # ---- Nadam (Nesterov + Adam) ------------------------------ #
            elif optimizer == 'nadam':
                if f'mW{i}' not in self.velocity:
                    self.velocity[f'mW{i}'] = np.zeros_like(dW)
                    self.velocity[f'mb{i}'] = np.zeros_like(db)
                    self.rms[f'vW{i}']      = np.zeros_like(dW)
                    self.rms[f'vb{i}']      = np.zeros_like(db)
                self.velocity[f'mW{i}'] = beta1 * self.velocity[f'mW{i}'] + (1 - beta1) * dW
                self.velocity[f'mb{i}'] = beta1 * self.velocity[f'mb{i}'] + (1 - beta1) * db
                self.rms[f'vW{i}']      = beta2 * self.rms[f'vW{i}']      + (1 - beta2) * (dW ** 2)
                self.rms[f'vb{i}']      = beta2 * self.rms[f'vb{i}']      + (1 - beta2) * (db ** 2)
                mW_hat = self.velocity[f'mW{i}'] / (1 - beta1 ** self.t)
                mb_hat = self.velocity[f'mb{i}'] / (1 - beta1 ** self.t)
                vW_hat = self.rms[f'vW{i}']      / (1 - beta2 ** self.t)
                vb_hat = self.rms[f'vb{i}']      / (1 - beta2 ** self.t)
                # Nesterov lookahead: replace m_hat with (beta1*m_hat + (1-beta1)*dW/bias_corr)
                bias_corr = 1 - beta1 ** self.t
                nW = beta1 * mW_hat + (1 - beta1) * dW / bias_corr
                nb = beta1 * mb_hat + (1 - beta1) * db / bias_corr
                self.params[f'W{i}'] -= lr * nW / (np.sqrt(vW_hat) + epsilon)
                self.params[f'b{i}'] -= lr * nb / (np.sqrt(vb_hat) + epsilon)

            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

    # ------------------------------------------------------------------ #
    #  One-hot encoder                                                     #
    # ------------------------------------------------------------------ #
    def one_hot(self, y, num_classes=10):
        m = y.shape[0]
        oh = np.zeros((m, num_classes))
        oh[np.arange(m), y] = 1
        return oh

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #
    def train(self, X, y, epochs=10, learning_rate=0.001, batch_size=64,
              optimizer='adam', weight_decay=0.0, verbose=True):
        """
        Args:
            X            : (N, 784) float array
            y            : (N,)     int labels 0-9
            epochs       : number of full passes
            learning_rate: step size
            batch_size   : mini-batch size
            optimizer    : 'sgd'|'momentum'|'nesterov'|'rmsprop'|'adam'|'nadam'
            weight_decay : L2 regularisation coefficient
            verbose      : print loss every 5 epochs
        Returns:
            history dict with 'loss' list
        """
        y_onehot = self.one_hot(y)
        history  = {'loss': []}

        for epoch in range(epochs):
            perm   = np.random.permutation(X.shape[0])
            X_shuf = X[perm]
            y_shuf = y_onehot[perm]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuf[i:i + batch_size]
                y_batch = y_shuf[i:i + batch_size]
                y_pred  = self.forward(X_batch)
                grads   = self.backward(y_batch)
                self.update_params(grads, lr=learning_rate, optimizer=optimizer,
                                   weight_decay=weight_decay)

            loss = self.compute_loss(self.forward(X), y_onehot)
            history['loss'].append(loss)

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

        return history

    # ------------------------------------------------------------------ #
    #  Predict                                                             #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Returns class indices (N,)"""
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
