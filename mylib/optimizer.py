from mylib import learning_parameter as lp
np = lp.xp_factory()


def set_optimizer(opt, lr):
    if opt == "SGD":
        return SGD(lr = lr)
    if opt == "Adam":
        return Adam(lr = lr)
    if opt == "AdaBound":
        return AdaBound(lr = lr, final_lr = 0.1)
    else:
        print("{} is not defined.".format(opt))
        return None


class SGD:
    def __init__(self, lr = 0.1):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Adam:
    def __init__(self, lr):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.iters = 0
        self.m = None
        self.v = None
        self.eps = 1e-7

    def update(self, params, grads):
        if self.m is None and self.v is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iters += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iters) / (1.0 - self.beta1**self.iters)
        for key in params.keys():
            self.m[key] += (1.0 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1.0 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)


class AdaBound:
    def __init__(self, lr, final_lr):
        self.lr = lr
        self.final_lr = final_lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.iters = 0
        self.m = None
        self.v = None
        self.eps = 1e-7

    def update(self, params, grads):
        if self.m is None and self.v is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iters += 1
        lower_lr = self.final_lr * (1.0 - 1.0 / ((1.0-self.beta2) * self.iters + 1.0))
        higher_lr = self.final_lr * (1.0 + 1.0 / ((1.0-self.beta2) * self.iters))
        for key in params.keys():
            self.m[key] += (1.0 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1.0 - self.beta2) * (grads[key]**2 - self.v[key])
            lr_t = np.clip(self.lr / (np.sqrt(self.v[key]) + self.eps), lower_lr, higher_lr) / np.sqrt(self.iters)
            params[key] -= lr_t * self.m[key]
