from esi_surrogate import forward
import numpy as np
import matplotlib.pyplot as plt


def main():
    # EXAMPLE USAGE
    # data format: [d, rc, alpha, h, ra] in each row. test_data.shape = (Nsamples, Nfeatures)
    test_data = np.loadtxt('geometry_samples.txt', skiprows=1)
    test_data[:, 2] = test_data[:, 2] * (np.pi / 180)  # convert to radians

    E_max = forward(test_data.T)
    plt.hist(E_max, density=True, bins=50, edgecolor='black')
    plt.show()


if __name__ == '__main__':
    main()
