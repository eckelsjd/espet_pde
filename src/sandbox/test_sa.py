from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import comb
from pathlib import Path

from src.scripts.esi_surrogate import forward
from src.scripts.geom_sampler import lhc_sampler


def get_sample_matrices(N, sampler):
    # Two independent sample matrices of params using latin hypercube
    A = sampler(N)
    B = sampler(N)
    N, d = A.shape

    # Combine matrices to get ABi and BAi
    AB = np.tile(A[:, :, np.newaxis], (1, 1, d))
    BA = np.tile(B[:, :, np.newaxis], (1, 1, d))
    for i in range(d):
        AB[:, i, i] = B[:, i]
        BA[:, i, i] = A[:, i]

    return A, B, AB, BA


def plot_sobol_indices(S1, ST, labels, conf=0.95):
    N, d = S1.shape
    z = st.norm.interval(alpha=conf)[1]  # get z-score from N(0,1), assuming CLT at n>30
    S1_est = np.mean(S1, axis=0)
    S1_se = np.sqrt(np.var(S1, axis=0) / N)
    ST_est = np.mean(ST, axis=0)
    ST_se = np.sqrt(np.var(ST, axis=0) / N)

    x = np.arange(d)
    width = 0.2

    plt.rc('font', size=12)
    fig, ax = plt.subplots()
    ax.bar(x - width/2, S1_est, width, color=[255/255, 253/255, 200/255], yerr=S1_se*z, label=r'$S_i$', capsize=5,
           linewidth=1, edgecolor=[0, 0, 0])
    ax.bar(x + width/2, ST_est, width, color=[131/255, 0/255, 38/255], yerr=ST_se*z, label=r'$S_{Ti}$', capsize=5,
           linewidth=1, edgecolor=[0, 0, 0])

    ax.set_ylabel(r"Sobol' index")
    ax.set_xlabel(r"Model parameters $\theta$")
    labels = [r'$d$', r'$R_c$', r'$\alpha$', r'$h$', r'$R_a$']
    ax.set_xticks(x, labels)
    ax.legend()
    plt.tight_layout()
    data_dir = Path('../../data/geometry')
    plt.savefig(str(data_dir / 'post' / 'sa_bar_chart.png'))
    plt.show()

    return fig, ax


def plot_pie_chart(S1, S2, labels):
    N, d = S1.shape
    S1_est = np.mean(S1, axis=0)
    S2_est = np.mean(S2, axis=0)

    fig_labels = labels.copy()
    sizes = S1_est.copy()

    # Append 2nd-order results
    for i in range(d):
        for j in range(i+1, d):
            fig_labels.append(f"({labels[i]}, {labels[j]})")
            sizes = np.append(sizes, S2_est[i, j])

    n = len(sizes)
    explode = np.zeros(n)*0.05
    # explode[1] = 0.1
    explode = tuple(explode)

    fig, ax = plt.subplots()
    ax.pie(np.abs(sizes), explode=explode, labels=fig_labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    data_dir = Path('../../data/geometry')
    plt.savefig(str(data_dir/'post'/'sa_pie_chart.png'))
    plt.show()

    return fig, ax


if __name__ == '__main__':
    # File locations
    data_dir = Path('../../data/geometry')
    net_file = str(data_dir / 'models' / 'esi_surrogate.onnx')
    data_file = str(data_dir / 'models' / 'norm_data.mat')

    # Problem-specific stuff
    def model(x):
        """Compute model output for inputs x
        :param x: (N, d) model inputs, N samples and d parameters
        :return: (N,) scalar model outputs
        """
        # A = 7
        # B = 0.1
        # return np.sin(x[:, 0]) + A * np.power(np.sin(x[:, 1]), 2) + B * np.power(x[:, 2], 4) * np.sin(x[:, 0])
        return forward(x.T, net_file=net_file, data_file=data_file, V0=1000)

    def sampler(N):
        """Sample model parameters
        :param N: Numer of samples to obtain
        :return: (N, d) samples of the d parameters
        """
        return lhc_sampler(N)
        # return lhc_sampler(N, reject=False, l_bounds=[-np.pi, -np.pi, -np.pi], u_bounds=[np.pi, np.pi, np.pi])

    # Get samples
    N = 100000  # Number of MC samples
    A, B, AB, BA = get_sample_matrices(N, sampler)
    d = A.shape[1]
    S1 = np.zeros((N, d))       # first-order
    S2 = np.zeros((N, d, d))    # second-order
    ST = np.zeros((N, d))       # total-order

    # Evaluate model; (d+2) * N evaluations
    fA = model(A)
    fB = model(B)
    fAB = np.zeros((N, d))
    fBA = np.zeros((N, d))
    for i in range(d):
        fAB[:, i] = model(AB[:, :, i])
        fBA[:, i] = model(BA[:, :, i])

    # Calculate sensitivity indices
    vY = np.var(np.r_[fA, fB])  # sample variance of output
    for i in range(d):
        S1[:, i] = fB * (fAB[:, i] - fA) / vY
        ST[:, i] = 0.5 * (fA - fAB[:, i]) ** 2 / vY

        # Second order
        for j in range(i+1, d):
            Vi = fB * (fAB[:, i] - fA)
            Vj = fB * (fAB[:, j] - fA)
            Vij = fBA[:, i] * fAB[:, j] - fA * fB
            S2[:, i, j] = (Vij - Vi - Vj) / vY

    # Construct confidence intervals
    confidence = 0.95  # 95% confidence level
    z = st.norm.interval(alpha=confidence)[1]  # get z-score from N(0,1), assuming CLT at n>30

    S1_est = np.mean(S1, axis=0)
    S1_se = np.sqrt(np.var(S1, axis=0) / N)
    S1_ci = [S1_est - z * S1_se, S1_est + z * S1_se]

    S2_est = np.mean(S2, axis=0)  # dxd matrix with row i, col j -> Sij
    S2_se = np.sqrt(np.var(S2, axis=0) / N)
    S2_ci = [S2_est - z * S2_se, S2_est + z * S2_se]

    ST_est = np.mean(ST, axis=0)
    ST_se = np.sqrt(np.var(ST, axis=0) / N)
    ST_ci = [ST_est - z * ST_se, ST_est + z * ST_se]

    # Print results
    names = ['d', 'rc', 'alpha', 'h', 'ra']
    # names = ['x1', 'x2', 'x3']
    print('%10s %10s %20s %10s %20s' % ('Param', 'S1_mean', 'S1_CI', 'ST_mean', 'ST_CI'))
    for i in range(d):
        print(f'{names[i]: >10} {S1_est[i]:10.5f} [{S1_ci[0][i]: ^8.5f}, {S1_ci[1][i]: ^8.5f}] {ST_est[i]:10.5f}'
              f' [{ST_ci[0][i]: ^8.5f}, {ST_ci[1][i]: ^8.5f}]')
    print(f"{'Total': >10} {np.sum(S1_est):10.5f} {' ':20} {np.sum(ST_est):10.5f} {' ':20}")
    print(' ')
    print('%10s %10s %20s' % ('2nd-order', 'S2_mean', 'S2_CI'))
    for i in range(d):
        for j in range(i+1, d):
            print(f"{names[i]: >4}-{names[j]: <4} {S2_est[i, j]: >10.5f}"
                  f" [{S2_ci[0][i, j]: >8.4f}, {S2_ci[1][i, j]: >8.4f}]")
    print(f"{'2nd-total': >10} {np.sum(S2_est):10.5f} {' ':20}")
    print(f"{'All-total': >10} {np.sum(S2_est) + np.sum(S1_est):10.5f} {' ':20}")
    print(' ')

    # Plot results\
    fig, ax = plot_sobol_indices(S1, ST, names)
    fig, ax = plot_pie_chart(S1, S2, names)

    # Save results to csv
    # S1 first-order indices
    rows = ['Mean', 'CI_lower', 'CI_upper']
    data = np.concatenate((S1_est[:, np.newaxis], S1_ci[0][:, np.newaxis], S1_ci[1][:, np.newaxis]), axis=1)
    df = pd.DataFrame(data.T, index=rows, columns=names)
    df.to_csv(data_dir/'post'/'S1_sobol_indices.csv')

    # S2 second-order indices
    data = np.zeros((3, int(comb(d, 2))))
    idx = 0
    headers = []
    for i in range(d):
        for j in range(i+1, d):
            headers.append(f'({names[i]}, {names[j]})')
            data[0, idx] = S2_est[i, j]
            data[1, idx] = S2_ci[0][i, j]
            data[2, idx] = S2_ci[1][i, j]
            idx += 1
    df = pd.DataFrame(data, index=rows, columns=headers)
    df.to_csv(data_dir / 'post' / 'S2_sobol_indices.csv')

    # ST total-order indices
    data = np.concatenate((ST_est[:, np.newaxis], ST_ci[0][:, np.newaxis], ST_ci[1][:, np.newaxis]), axis=1)
    df = pd.DataFrame(data.T, index=rows, columns=names)
    df.to_csv(data_dir / 'post' / 'ST_sobol_indices.csv')
