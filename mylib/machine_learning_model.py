from mylib.layers import *
from mylib.numerical_gradient import *
from collections import OrderedDict
from mylib import learning_parameter as lp
np = lp.xp_factory()

class MachineLearningModel:
    def __init__(self, input_size, input_dim,
                 hidden, act_func, weight_init, batch_norm, batch_norm_output,
                 output_size, output_dim, lastlayer_identity,
                 loss_func, weight_decay, decay_lambda,
                 acc_func = "RelativeError", dtype = np.float32):
        self.input_size                                 = input_size
        self.input_dim                                  = input_dim
        self.hidden, self.act_func, self.weight_init    = hidden, act_func, weight_init
        self.batch_norm, self.batch_norm_output         = batch_norm, batch_norm_output
        self.output_size, self.output_dim               = output_size, output_dim
        self.lastlayer_identity                         = lastlayer_identity
        self.loss_func                                  = loss_func
        self.weight_decay, self.decay_lambda            = weight_decay, decay_lambda
        self.dtype                                      = dtype

        self.params         = {}
        self.__init_weight(self.weight_init)
        self.input_layers   = {}
        self.layers         = OrderedDict()
        self.output_layers  = {}
        self.loss_layers    = {}
        self.__init_layers()
        self.acc_func       = acc_func


    def __init_weight(self, w_init):
        ## Initialize input layer's weight.
        for dim in range(1, self.input_dim+1):
            if w_init == "he":
                scale = np.sqrt(2.0 / self.input_size)
            elif w_init == "xavier":
                scale = np.sqrt(1.0 / self.input_size)
            else:
                print("weight_init is not defined.\nScale will be set as 1.0")
                scale = 1.0

            self.params["Weight_input{}".format(dim)]   = scale * np.random.randn(self.input_size * 2, self.hidden[0])
            self.params["Bias_input{}".format(dim)]     = np.zeros(self.hidden[0])
            if self.batch_norm:
                self.params["Gamma_input{}".format(dim)]    = np.ones(self.hidden[0])
                self.params["Beta_input{}".format(dim)]     = np.zeros(self.hidden[0])

        ## Initialize hidden layer's weight.
        for i in range(1, len(self.hidden)):
            if w_init == "he":
                scale = np.sqrt(2.0 / self.hidden[i-1])
            elif w_init == "xavier":
                scale = np.sqrt(1.0 / self.hidden[i-1])
            else:
                print("weight_init is not defined.\nScale will be set as 1.0")
                scale = 1.0

            self.params["Weight{}".format(i)]   = scale * np.random.randn(self.hidden[i-1], self.hidden[i])
            self.params["Bias{}".format(i)]     = np.zeros(self.hidden[i])
            if self.batch_norm:
                self.params["Gamma{}".format(i)]    = np.ones(self.hidden[i])
                self.params["Beta{}".format(i)]     = np.zeros(self.hidden[i])

        ## Initialize output layer's weight.
        for dim in range(1, self.output_dim+1):
            if w_init == "he":
                scale = np.sqrt(2.0 / self.hidden[-1])
            elif w_init == "xavier":
                scale = np.sqrt(1.0 / self.hidden[-1])
            else:
                print("weight_init is not defined.\nScale will be set as 1.0")
                scale = 1.0

            self.params["Weight_output{}".format(dim)]  = scale * np.random.randn(self.hidden[-1], self.output_size)
            self.params["Bias_output{}".format(dim)]    = np.zeros(self.output_size)
            if self.batch_norm and self.batch_norm_output:
                self.params["Gamma_output{}".format(dim)]   = np.ones(self.output_size)
                self.params["Beta_output{}".format(dim)]    = np.zeros(self.output_size)

        ## Set data type.
        for key in self.params.keys():
            self.params[key] = self.params[key].astype(self.dtype)


    def __init_layers(self):
        ## Make input layers.
        for dim in range(1, self.input_dim+1):
            self.input_layers[dim]                                      = OrderedDict()
            self.input_layers[dim]["Affine_input{}".format(dim)]        = Affine(self.params["Weight_input{}".format(dim)], self.params["Bias_input{}".format(dim)])
            if self.batch_norm:
                self.input_layers[dim]["BatchNorm_input{}".format(dim)] = BatchNormalization(self.params["Gamma_input{}".format(dim)], self.params["Beta_input{}".format(dim)])
            self.input_layers[dim]["Activation_input{}".format(dim)]    = activation_function(self.act_func)

        ## Make hidden layers.
        for i in range(1, len(self.hidden)):
            self.layers["Affine{}".format(i)]           = Affine(self.params["Weight{}".format(i)], self.params["Bias{}".format(i)])
            if self.batch_norm:
                self.layers["BatchNorm{}".format(i)]    = BatchNormalization(self.params["Gamma{}".format(i)], self.params["Beta{}".format(i)])
            self.layers["Activation{}".format(i)]       = activation_function(self.act_func)

        ## Make output layers.
        for dim in range(1, self.output_dim+1):
            self.output_layers[dim]                                         = OrderedDict()
            self.output_layers[dim]["Affine_output{}".format(dim)]          = Affine(self.params["Weight_output{}".format(dim)], self.params["Bias_output{}".format(dim)])
            if self.batch_norm and self.batch_norm_output:
                self.output_layers[dim]["BatchNorm_output{}".format(dim)]   = BatchNormalization(self.params["Gamma_output{}".format(dim)], self.params["Beta_output{}".format(dim)])
            if self.lastlayer_identity:
                self.output_layers[dim]["Activation_output{}".format(dim)]  = Identity()
            else:
                self.output_layers[dim]["Activation_output{}".format(dim)]  = activation_function(self.act_func)
            ## Make loss layers.
            self.loss_layers[dim]                                           = loss_function(self.loss_func)


    def predict(self, x, is_training):
        ## Forwarding in input layers.
        out = 0
        for dim in range(1, self.input_dim+1):
            for layer_name, layer in self.input_layers[dim].items():
                if layer_name == list(self.input_layers[dim].keys())[0]:
                    out_ = layer.forward(x[dim-1, ...], is_training)
                else:
                    out_ = layer.forward(out_, is_training)
            out += out_
        out /= self.input_dim

        ## Forwarding in hidden layers.
        for layer in self.layers.values():
            out = layer.forward(out, is_training)

        ## Forwarding in output layers.
        out_list = []
        for dim in range(1, self.output_dim+1):
            for layer_name, layer in self.output_layers[dim].items():
                if layer_name == list(self.output_layers[dim].keys())[0]:
                    out_ = layer.forward(out, is_training)
                else:
                    out_ = layer.forward(out_, is_training)
            out_list.append(out_)
        out_list = np.array(out_list)

        return out_list


    def loss(self, x, t, is_training):
        y = self.predict(x, is_training)

        loss_list = []
        for dim in range(1, self.output_dim+1):
            loss = self.loss_layers[dim].forward(y[dim-1, ...], t[dim-1, ...])
            loss_list.append(loss)

        if self.weight_decay:
            norm = 0
            for dim in range(1, self.input_dim+1):
                norm += self.decay_lambda * np.sum(self.params["Weight_input{}".format(dim)]**2) / 2.0

            for i in range(1, len(self.hidden)):
                norm += self.decay_lambda * np.sum(self.params["Weight{}".format(i)]**2) / 2.0

            for dim in range(1, self.output_dim+1):
                norm_ = norm + self.decay_lambda * np.sum(self.params["Weight_output{}".format(dim)]**2) / 2.0
                loss_list[dim-1] += norm_

        loss_list = np.array(loss_list)
        return loss_list


    def accuracy(self, x, t, acc_func, is_training):
        y = self.predict(x, is_training)
        acc_list = []
        if self.acc_func == "RelativeError":
            for dim in range(self.output_dim):
                mask = (t[dim, ...] == 0.0)
                t[dim, mask] += 1e-7
                y[dim, mask] += 1e-7
                error = (y[dim, ...] - t[dim, ...]) / t[dim, ...]
                acc = np.mean(np.abs(error))
                acc_list.append(acc)
        else:
            print("Accuracy function is not defined.")

        return np.array(acc_list)


    def gradient(self, x, t, is_training = True):
        ## Forwarding.
        self.loss(x, t, is_training)

        ## Backwarding from lastlayer to output layer.
        dout = 0.0
        for dim in range(1, self.output_dim+1):
            dout_   = self.loss_layers[dim].backward(dout = 1.0)
            layers  = list(self.output_layers[dim].values())
            layers.reverse()
            for layer in layers:
                dout_ = layer.backward(dout_)
            dout    += dout_

        ## Backwarding in hidden layers.
        layers  = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        dout    /= self.input_dim
        ## Backwarding in input layers.
        for dim in range(1, self.input_dim+1):
            layers  = list(self.input_layers[dim].values())
            keys    = list(self.input_layers[dim].keys())
            layers.reverse()
            keys.reverse()
            for key, layer in zip(keys, layers):
                if layer == layers[0]:
                    dout_ = layer.backward(dout)
                else:
                    dout_ = layer.backward(dout_)

        ## Get gradients of self.params.
        ## Input layers.
        grads = {}
        for dim in range(1, self.input_dim+1):
            grads["Weight_input{}".format(dim)]     = self.input_layers[dim]["Affine_input{}".format(dim)].dW
            grads["Bias_input{}".format(dim)]       = self.input_layers[dim]["Affine_input{}".format(dim)].db
            if self.batch_norm:
                grads["Gamma_input{}".format(dim)]  = self.input_layers[dim]["BatchNorm_input{}".format(dim)].dgamma
                grads["Beta_input{}".format(dim)]   = self.input_layers[dim]["BatchNorm_input{}".format(dim)].dbeta

        ## Hidden layers.
        for i in range(1, len(self.hidden)):
            grads["Weight{}".format(i)]     = self.layers["Affine{}".format(i)].dW
            grads["Bias{}".format(i)]       = self.layers["Affine{}".format(i)].db
            if self.batch_norm:
                grads["Gamma{}".format(i)]  = self.layers["BatchNorm{}".format(i)].dgamma
                grads["Beta{}".format(i)]   = self.layers["BatchNorm{}".format(i)].dbeta

        ## Output layers.
        for dim in range(1, self.output_dim+1):
            grads["Weight_output{}".format(dim)]    = self.output_layers[dim]["Affine_output{}".format(dim)].dW
            grads["Bias_output{}".format(dim)]      = self.output_layers[dim]["Affine_output{}".format(dim)].db
            if self.batch_norm and self.batch_norm_output:
                grads["Gamma_output{}".format(dim)] = self.output_layers[dim]["BatchNorm_output{}".format(dim)].dgamma
                grads["Beta_output{}".format(dim)]  = self.output_layers[dim]["BatchNorm_output{}".format(dim)].dbeta

        if self.weight_decay:
            for key in grads.keys():
                if "Weight" in key:
                    grads[key] += self.decay_lambda * self.params[key]

        for key, grad in grads.items():
            grads[key] = grad.astype(self.dtype)

        return grads


    def numerical_gradient(self, x, t):
        ## For debugging the backpropagation.
        loss_W  =lambda W: self.loss(x, t, is_training = True)
        grads   = {}
        for key, param in self.params.items():
            grads[key] = numerical_gradient(loss_W, param, self.output_dim)
            grads[key] = grads[key].astype(self.dtype)

        return grads
