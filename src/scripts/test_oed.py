# import math
import numpy as np
from numba import jit, prange
# from scipy.stats import norm
import matplotlib.pyplot as plt


# Issue: https://github.com/numba/numba/issues/1269
@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


# Workaround to add numba compatible axis=1 argument to mean
@jit(nopython=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


# Simple nonlinear model example with thetad=1, xd=1
def nonlinear_model(theta, x):
    thetad, Ns = theta.shape
    xd, Nx = x.shape
    assert thetad == 1 and xd == 1

    return np.square(x).T @ np.power(theta, 3) + np.exp(-abs(0.2-x)).T @ theta


@jit(nopython=True)
def gaussian_1d(x, mu, var):
    # return norm.pdf(x, loc=mu, scale=var)
    return 1/np.sqrt(2*np.pi*var) * np.exp((-0.5/var)*np.square(x-mu))


@jit(nopython=True, parallel=True)
def compute_utility(y, g_theta, noise_var):
    Nx, Ns = y.shape
    likelihood = gaussian_1d(y, g_theta, noise_var)  # (Nx, Ns) grid
    evidence = np.zeros((Nx, Ns), dtype=np.float32)
    for j in prange(Ns):
        print(f'Loop j: {j} out of {Ns - 1}')
        y_j = y[:, j]
        y_tile = y_j.repeat(Ns).reshape((-1, Ns))  # jit compatible
        # y_tile = np.tile(y_j[:, np.newaxis], (1, Ns))
        # marginal_like = np.mean(gaussian_1d(y_tile, g_theta, noise_var), axis=1)
        marginal_like = np_mean(gaussian_1d(y_tile, g_theta, noise_var), axis=1)
        evidence[:, j] = marginal_like

    # Expected information gain
    # utility = np.mean(np.log(likelihood) - np.log(evidence), axis=1)  # (Nx,)
    utility = np_mean(np.log(likelihood) - np.log(evidence), axis=1)  # (Nx,)

    return utility


# Assumes Gaussian noise model: y(theta, x) = G(theta, x) + epsilon,  where epsilon ~ N(0, var)
# y(theta,x) should be a single scalar output
def expected_information(Ns, Nx, x_sampler, prior_sampler, model, noise_var=1e-4):
    def fix_shape(x):
        x = np.atleast_1d(np.squeeze(x)).astype(np.float32)
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        return x

    # Sample experimental input locations x
    x_samples = fix_shape(x_sampler(Nx))  # (xd, Nx)
    xd = x_samples.shape[0]

    # Sample the prior
    theta_samples = fix_shape(prior_sampler(Ns))  # (thetad, Ns)
    thetad = theta_samples.shape[0]

    # Evaluate the model
    g_theta = fix_shape(model(theta_samples, x_samples))  # (Nx, Ns)
    assert g_theta.shape == (Nx, Ns)

    # Get samples of y
    y = np.random.normal(loc=g_theta, scale=np.sqrt(noise_var)).astype(np.float32)  # (Nx, Ns)

    utility = compute_utility(y, g_theta, np.array(noise_var))
    return x_samples, utility


if __name__ == '__main__':
    x_sampler = lambda Nx: np.linspace(0, 1, Nx)
    theta_sampler = lambda Ns: np.random.rand(Ns)
    Ns = 30000
    Nx = 101
    d, U_d = expected_information(Ns, Nx, x_sampler, theta_sampler, nonlinear_model)
    plt.figure()
    plt.plot(np.squeeze(d), U_d, '-k')
    plt.xlabel('$d$')
    plt.ylabel('$\hat{U}(d)$')
    plt.show()

    # Tile and flip g_theta's to align with repeated y's on 3rd axis
    # y_tile = np.tile(y[:,:,np.newaxis],(1,1,Ns)) # (Nx, Ns, Ns)
    # g_theta_tile = np.tile(g_theta[:, np.newaxis, :], (1, Ns, 1)) # (Nx, Ns, Ns)
    # Evidence
    # evidence = np.mean(gaussian_1d(y_tile, g_theta_tile, noise_var), axis=2)  # (Nx, Ns)
    # Too big to fit single (Nx, Ns, Ns) in memory, break up 2nd axis
    # bs = 500  # batch size
    # for j in range(math.ceil(Ns/bs)):
    #     print(f'Loop j: {j} out of {math.ceil(Ns/bs)-1}')
    #     start_idx = j*bs
        # end_idx = min(start_idx + bs, Ns)
        # y_j = y[:, start_idx:end_idx]  # (Nx, bs)
        # y_tile = np.tile(y_j[:, :, np.newaxis], (1, 1, Ns))  # (Nx, bs, Ns)
        # g_theta_tile = np.tile(g_theta[:, np.newaxis, :], (1, bs, 1))  # (Nx, bs, Ns)
        # like = gaussian_1d(y_tile, g_theta_tile, noise_var)
        # marginal_like = np.mean(like, axis=2)
        # evidence[:, start_idx:end_idx] = marginal_like  # (Nx, bs)
        # y_tile = np.tile(y_j[:,np.newaxis], (1, Ns))
        # evidence[:, start_idx:end_idx] = np.mean(gaussian_1d(y_tile, g_theta, noise_var), axis=1) # (Nx, Ns)
