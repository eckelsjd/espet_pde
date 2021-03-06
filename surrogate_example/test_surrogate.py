from esi_surrogate import forward
import numpy as np
import matplotlib.pyplot as plt


def main():
    # EXAMPLE USAGE
    # data format: [d, rc, alpha, h, ra] in each row. test_data.shape = (Nsamples, Nfeatures)
    test_data = np.loadtxt('geometry_samples.txt', skiprows=1)
    test_data[:, 2] = test_data[:, 2] * (np.pi / 180)  # convert to radians

    # input to forward() shape = (Nfeatures, Nsamples), where Nfeatures = 5
    E_max = forward(test_data.T)  # E_max.shape = (Nsamples,)
    plt.hist(E_max, density=True, bins=50, edgecolor='black')
    plt.xlabel('Electric field magnitude [V/m]')
    plt.ylabel('Empirical PDF over test set')
    plt.show()


if __name__ == '__main__':
    main()
