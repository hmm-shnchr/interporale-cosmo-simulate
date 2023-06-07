from mylib import learning_parameter as lp
np = lp.xp_factory()


def numerical_gradient(f, x, output_dim):
    h = 1e-4
    grad = np.zeros_like(x)

    if x.ndim == 1:
        N = x.shape[0]
        for n in range(N):
            for i in range(output_dim):
                tmp_val = x[n]
                x[n] = tmp_val + h
                fxh1 = f(x)
                fxh1 = fxh1[i]
                x[n] = tmp_val - h
                fxh2 = f(x)
                fxh2 = fxh2[i]
                grad[n] = (fxh1 - fxh2) / (2.0 * h)
                x[n] = tmp_val

    elif x.ndim == 2:
        N, M = x.shape[0], x.shape[1]
        for n in range(N):
            for m in range(M):
                for i in range(output_dim):
                    tmp_val = x[n, m]
                    x[n, m] = tmp_val + h
                    fxh1 = f(x)
                    fxh1 = fxh1[i]
                    x[n, m] = tmp_val - h
                    fxh2 = f(x)
                    fxh2 = fxh2[i]
                    grad[n, m] = (fxh1 - fxh2) / (2.0 * h)
                    x[n, m] = tmp_val

    return grad
