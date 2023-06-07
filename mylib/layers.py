from mylib import learning_parameter as lp
np = lp.xp_factory()


def activation_function(act_func):
    if act_func == "relu":      return Relu()
    if act_func == "sigmoid":   return Sigmoid()
    if act_func == "tanh":      return Tanh()
    if act_func == "mish":      return Mish()
    if act_func == "tanhexp":   return TanhExp()
    else:
        print("{} is not defined.".format(act_func))
        return None


def loss_function(loss_func):
    if loss_func == "MSE_RE":       return MSE_RelativeError()
    if loss_func == "MSE_AE":       return MSE_AbsoluteError()
    if loss_func == "RE":           return RelativeError()
    if loss_func == "AE":           return AbsoluteError()
    if loss_func == "Log_RE":       return Log_RelativeError()
    if loss_func == "Sqrt_RE":      return Sqrt_RelativeError()
    if loss_func == "Sqrt_AE":      return Sqrt_AbsoluteError()
    if loss_func == "RE_to_MSE_RE": return RE_to_MSE_RE()
    else:
        print("{} is not defined.".format(loss_func))
        return None


class Affine:
    def __init__(self, W, b):
        self.W  = W
        self.b  = b
        self.x  = None
        self.dW = None
        self.db = None

    def forward(self, x, is_training):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx      = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx


class Embedding:
    def __init__(self, W):
        self.W                  = W
        self.x                  = None
        self.idx                = None
        self.N, self.D, self.H  = None, None, None
        self.W_                 = None
        self.dW                 = None

    def forward(self, x, idx, is_training):
        self.N, self.D, self.H  = x.shape[0], x.shape[1], self.W.shape[1]
        self.x                  = x
        self.idx                = idx
        self.W_                 = np.empty((self.N, self.D, self.H))
        out = np.empty((self.N, self.H))
        for n in range(self.N):
            self.W_[n]  = self.W[idx[n]]
            x_          = x[n].reshape(1, -1)
            out[n]      = np.dot(x_, self.W_[n]).reshape(-1)
        return out

    def backward(self, dout):
        dx      = np.empty(self.x.shape)
        self.dW = np.zeros_like(self.W)
        for n in range(self.N):
            dout_                   = dout[n].reshape(1, -1)
            dx_                     = np.dot(dout_, self.W_[n].T)
            dx[n]                   = dx_.reshape(-1)
            x_                      = self.x[n].reshape(-1, 1)
            x_                      = x_.T.reshape(1, -1)
            dW_                     = np.dot(x_.T, dout_)
            self.dW[self.idx[n]]    += dW_
        return dx


def Smoothing(x, input_x, input_size, output_size):
    conc_x = np.concatenate([input_x[:, :input_size], x, input_x[:, input_size:]], axis = 1)
    for i in range(input_size, input_size + output_size):
        conc_x[:, i] = np.sum(conc_x[:, i-input_size:i+input_size+1], axis = 1) / (input_size * 2 + 1)
    return conc_x[:, input_size:input_size+output_size]

def Smoothing_2(x, input_x, input_size, output_size):
    conc_x = np.concatenate([input_x[:, :input_size], x, input_x[:, input_size:]], axis = 1)
    out     = np.empty(x.shape)
    for i in range(input_size, input_size + output_size):
        out[:, i-input_size] = np.sum(conc_x[:, i-input_size:i+input_size+1], axis = 1) / (input_size * 2 + 1)
    return out


class Smoothing_proto:
    def __init__(self, input_size, output_size):
        self.input_size     = input_size
        self.output_size    = output_size
        self.dout_          = None

    def forward(self, x, input_x):
        conc_x  = np.concatenate([input_x[:, :self.input_size], x, input_x[:, self.input_size:]], axis = 1)
        out     = np.empty(x.shape)
        for i in range(self.input_size, self.input_size + self.output_size):
            out[:, i-self.input_size] = np.sum(conc_x[:, i-self.input_size:i+self.input_size+1], axis = 1) / (self.input_size * 2 + 1)
        return out

    def backward(self, dout):
        if self.dout_ is None:
            dout_       = np.empty(dout.shape)
            numerator   = self.input_size * 2 + 1
            denominator = self.input_size * 2 + 1
            for i in range(self.output_size):
                if i < self.input_size:
                    dout_[:, i] = (numerator + (i - self.input_size)) / denominator
                elif i + self.input_size > self.output_size - 1:
                    dout_[:, i] = (numerator - (i + self.input_size - self.output_size + 1)) / denominator
                else:
                    dout_[:, i] = numerator / denominator
            self.dout_ = dout_
        dout *= self.dout_
        return dout


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x, is_training):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x, is_training):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x, is_training):
        #self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out**2)


class Mish:
    def __init__(self):
        self.x      = None
        self.expx   = None

    def forward(self, x, is_training):
        self.x      = x
        self.expx   = np.exp(x)
        return self.x * np.tanh(np.log(1.0 + self.expx))

    def backward(self, dout):
        diff    = 4.0 * (self.x + 1.0 + self.expx**2) + self.expx**3 + (4.0 * self.x + 6.0) * self.expx
        diff    *= self.expx
        diff    /= (2.0 * self.expx + self.expx**2 + 2.0)**2
        return dout * diff


class TanhExp:
    def __init__(self):
        self.x = None

    def forward(self, x, is_training):
        self.x = x
        return x * np.tanh(np.exp(x))

    def backward(self, dout):
        return dout * (np.tanh(np.exp(self.x)) - self.x * np.exp(self.x) * (np.tanh(np.exp(self.x))**2 - 1.0))


class Identity:
    def forward(self, x, is_training):
        return x

    def backward( self, dout ):
        return dout


class MSE_RelativeError:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y, self.t = y, t
        error = (y - t) / t
        return np.mean(error**2)

    def backward(self, dout = 1.0):
        return dout * (2.0 * (self.y - self.t) / self.t**2) / float(self.y.size)


class MSE_AbsoluteError:
    def __init__(self):
        self.y  = None
        self.t  = None

    def forward(self, y, t):
        self.y  = y
        self.t  = t
        error   = (y - t)
        return np.mean(error**2)

    def backward(self, dout = 1):
        return dout * 2.0 * (self.y - self.t) / float(self.y.size)


class RelativeError:
    def __init__(self):
        self.t      = None
        self.mask   = None

    def forward(self, y, t):
        self.t      = t
        error       = (y - t) / t
        self.mask   = (error < 0.0)
        return np.mean(np.abs(error))

    def backward(self, dout = 1.0):
        dout            /= self.t * self.t.size
        dout[self.mask] *= -1.0
        return dout


class AbsoluteError:
    def __init__(self):
        self.mask   = None
        self.shape  = None
        self.size   = None

    def forward(self, y, t):
        error       = y - t
        self.mask   = (error < 0.0)
        self.shape  = error.shape
        self.size   = error.size
        return np.mean(np.abs(error))

    def backward(self, dout = 1.0):
        dout            *= np.ones(self.shape) / float(self.size)
        dout[self.mask] *= -1.0
        return dout


class Log_RelativeError:
    def __init__(self):
        self.y  = None
        self.t  = None

    def forward(self, y, t):
        self.y  = y
        self.t  = t
        error   = (y - t) / t
        return np.mean(np.log(np.abs(error)))

    def backward(self, dout = 1.0):
        dout    /= (self.y - self.t) * self.y.size
        return dout


class Sqrt_RelativeError:
    def __init__(self):
        self.y      = None
        self.t      = None
        self.mask   = None

    def forward(self, y, t):
        self.y      = y
        self.t      = t
        error       = (y - t) / t
        self.mask   = (error < 0.0)
        return 2.0 * np.mean(np.sqrt(np.abs(error)))

    def backward(self, dout = 1.0):
        dout            /= float(self.y.size) * np.sqrt(np.abs((self.y - self.t) / self.t)) * self.t
        dout[self.mask] *= -1.0
        return dout


class Sqrt_AbsoluteError:
    def __init__(self):
        self.error  = None
        self.mask   = None

    def forward(self, y, t):
        self.mask   = ((y - t) < 0.0)
        self.error  = np.sqrt(np.abs(y - t))
        return 2.0 * np.mean(self.error)

    def backward(self, dout = 1.0):
        dout            /= float(self.error.size) * self.error
        dout[self.mask] *= -1.0
        return dout


class RE_to_MSE_RE:
    def __init__(self):
        self.y              = None
        self.t              = None
        self.mask_border    = None

    def forward(self, y, t):
        self.y = y
        self.t = t
        error = (y - t) / t
        self.mask_border = (np.abs(error) < 1.0)
        error[self.mask_border] = error[self.mask_border]**2 / 2.0
        error = np.abs(error)
        error[self.mask_border == False] -= 0.5
        return np.mean(np.abs(error))

    def backward(self, dout = 1.0):
        dout                            *= np.ones_like(self.y)
        dout[self.mask_border == False] /= self.t[self.mask_border == False]
        error                           = (self.y - self.t) / self.t
        mask                            = (error == 0.0)
        error[mask]                     += 1e-7
        dout                            *= error / np.abs(error)
        dout[self.mask_border == True]  = (self.y[self.mask_border == True] - self.t[self.mask_border == True]) / self.t[self.mask_border == True]**2
        dout                            /= float(self.y.size)
        return dout


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma          = gamma
        self.beta           = beta
        self.momentum       = momentum
        self.input_shape    = None
        self.running_mean   = running_mean
        self.running_var    = running_var
        self.batch_size     = None
        self.xc             = None
        self.std            = None
        self.dgamma         = None
        self.dbeta          = None

    def forward(self, x, is_training):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, is_training)

        return out.reshape(*self.input_shape)

    def __forward(self, x, is_training):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean   = np.zeros(D)
            self.running_var    = np.zeros(D)

        if is_training:
            mu  = x.mean(axis=0)
            xc  = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn  = xc / std

            self.batch_size = x.shape[0]
            self.xc             = xc
            self.xn             = xn
            self.std            = std
            self.running_mean   = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var    = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc  = x - self.running_mean
            xn  = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta   = dout.sum(axis=0)
        dgamma  = np.sum(self.xn * dout, axis=0)
        dxn     = self.gamma * dout
        dxc     = dxn / self.std
        dstd    = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar    = 0.5 * dstd / self.std
        dxc     += (2.0 / self.batch_size) * self.xc * dvar
        dmu     = np.sum(dxc, axis=0)
        dx      = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta  = dbeta

        return dx