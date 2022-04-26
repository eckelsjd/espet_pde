# FUCK numba nopython mode
# Issue: https://github.com/numba/numba/issues/1269
# 3D case is incredibly slow, I apologize for even including it here
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

    # if ndim == 1:
    #     raise Exception('Dont use')
    #     # result = func1d(arr)
    # elif ndim == 2:
    #     if axis == 0:
    #         result = np.empty(arr.shape[1])
    #         for i in range(len(result)):
    #             result[i] = func1d(arr[:, i])
    #     elif axis == 1:
    #         result = np.empty(arr.shape[0])
    #         for i in range(len(result)):
    #             result[i] = func1d(arr[i, :])
    #     else:
    #         raise Exception('Not enough axes')
    # elif ndim == 3:
    #     raise Exception('Please dont use')
        # if axis == 0:
        #     result = np.empty((arr.shape[1], arr.shape[2]))
        #     for i in range(arr.shape[1]):
        #         for j in range(arr.shape[2]):
        #             result[i, j] = func1d(arr[:, i, j])
        # elif axis == 1:
        #     result = np.empty((arr.shape[0], arr.shape[2]))
        #     for i in range(arr.shape[0]):
        #         for j in range(arr.shape[2]):
        #             result[i, j] = func1d(arr[i, :, j])
        # elif axis == 2:
        #     result = np.empty((arr.shape[0], arr.shape[1]))
        #     for i in range(arr.shape[0]):
        #         for j in range(arr.shape[1]):
        #             result[i, j] = func1d(arr[i, j, :])
        # else:
        #     raise Exception('Not enough axes')
    # else:
    #     raise Exception('Not implemented')

# Workaround to add numba compatible axis argument
@jit(nopython=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


# @jit(nopython=True)
# def np_prod(array, axis):
#     return np_apply_along_axis(np.prod, axis, array)

# if bs > 1:
#     # Product over batch size (numba won't allow np.prod(axis=0))
#     # likelihood = np_prod(likelihood, axis=0) # and this np_prod workaround sucks
#     like = likelihood[0, :, :]
#     for b in range(bs-1):
#         like = np.multiply(like, likelihood[b + 1, :, :])
#     likelihood = like
# else:
#     likelihood = np.reshape(likelihood, (Nx, Ns))

## EXTRA STUFF
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
#     end_idx = min(start_idx + bs, Ns)
#     y_j = y[:, start_idx:end_idx]  # (Nx, bs)
#     y_tile = np.tile(y_j[:, :, np.newaxis], (1, 1, Ns))  # (Nx, bs, Ns)
#     g_theta_tile = np.tile(g_theta[:, np.newaxis, :], (1, bs, 1))  # (Nx, bs, Ns)
#     like = gaussian_1d(y_tile, g_theta_tile, noise_var)
#     marginal_like = np.mean(like, axis=2)
#     evidence[:, start_idx:end_idx] = marginal_like  # (Nx, bs)
#     y_tile = np.tile(y_j[:,np.newaxis], (1, Ns))
#     evidence[:, start_idx:end_idx] = np.mean(gaussian_1d(y_tile, g_theta, noise_var), axis=1) # (Nx, Ns)