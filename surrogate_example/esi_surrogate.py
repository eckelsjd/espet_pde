# Run surrogate inference on input emitter geometry -> max electric field

# EXAMPLE USAGE:
# as script: python esi_surrogate.py
# as import: from esi_surrogate import forward

import numpy as np
import scipy.io  # to load .mat data
import onnx      # to load trained neural network
import onnxruntime as ort  # to run inference
from pathlib import Path
import matplotlib.pyplot as plt

# Hardcoded normalization settings from training data
# Network inputs
XMAX_INPUT = np.array([[3.90693257], [0.5518092], [1.04718602], [1.85828436]])
XMIN_INPUT = np.array([[1.61198281e-04], [-2.90936128e+00], [2.61868842e-01], [6.62629126e-03]])

# Network outputs
XMAX_OUTPUT = np.array([[2.45369903]])
XMIN_OUTPUT = np.array([[-0.25092102]])

# Normalization output range
YMIN = -1
YMAX = 1

# Bias Voltage
V0_DEFAULT = 1000  # V


def mapminmax(x, xmin, xmax, ymin, ymax, direction='Forward'):
    """
    Pre and post process normalization for FCNN. Replicates Matlab's mapminmax function.
    Ni: number of inputs
    No: number of outputs
    Nd: number of data samples
    :param x: (Ni,Nd) if 'Forward', (No,Nd) if 'Reverse', data that is to be normalized
    :param xmin: (Ni,1) if 'Forward', (No,1) if 'Reverse': minimum x value from NN training
    :param xmax: (Ni,1) if 'Forward', (No,1) if 'Reverse': maximum x value from NN training
    :param ymin: (1,1) mapping minimum for both NN inputs/outputs (assume they use the same mapping)
    :param ymax: (1,1) mapping maximum for both NN inputs/outputs (assume they use the same mapping)
    :param direction: 'Forward' for network inputs, 'Reverse' for network outputs
    :return:
    """
    if direction == 'Forward':
        # Apply the mapping x -> xnorm
        Ni, Nd = x.shape
        xmin_tile = np.tile(xmin, (1,Nd))
        xmax_tile = np.tile(xmax, (1,Nd))
        xnorm = np.divide((x - xmin_tile),(xmax_tile-xmin_tile))*(ymax-ymin) + ymin
        return xnorm
    elif direction == 'Reverse':
        # Undo the mapping y -> ynorm (equivalently, apply the mapping ynorm -> y)
        No, Nd = x.shape
        ynorm = x  # raw network output is already normalized
        ynorm_min = ymin
        ynorm_max = ymax
        ymin_tile = np.tile(xmin,(1,Nd))
        ymax_tile = np.tile(xmax,(1,Nd))
        ratio = (ynorm - ynorm_min)/(ynorm_max-ynorm_min)
        y = np.multiply(ratio, (ymax_tile - ymin_tile)) + ymin_tile
        return y
    else:
        print(f'Incorrect direction: {direction}')


def forward(x, net_file='esi_surrogate.onnx', data_file=None, V0=V0_DEFAULT):
    """Run a forward inference pass of the network saved in net_file on the inputs x.
    :param x: (5, Nsamples) np array where each column is an emitter geometry [d, Rc, alpha, h, Ra]^T
                in units of [m, m, rad, m, m]
    :param net_file: Path to the .onnx trained network file
    :param data_file: Path to norm_data.mat file containing normalization settings
    :param V0: (Nsamples,) or (1,) bias voltage for each geometry. If size is (1,), same bias applied to all geometries
    :return (1, Nsamples) E_max [V/m] predictions for each of the Nsamples geometry inputs
    """
    # NONDIMENSIONALIZE
    d = x[0, :]       # Tip-to-extractor distance [m]
    rc = x[1, :]      # Radius of curvature [m]
    alpha = x[2, :]   # Cone half-angle [rad]
    h = x[3, :]       # Emitter height [m]
    ra = x[4, :]      # Radius of aperture [m]

    d_tilde = np.divide(d, h)
    rc_tilde = np.log10(np.divide(rc, h))
    ra_tilde = np.divide(ra, h)
    x_nondim = np.vstack((d_tilde, rc_tilde, alpha, ra_tilde))

    # NORMALIZE INPUTS
    if data_file:
        # Load normalization settings from .mat file if specified
        data = scipy.io.loadmat(data_file)

        # Parse normalization settings from Matlab "process settings" struct
        xs = data['xs'][0][0]
        xmax = xs[2]
        xmin = xs[3]
        ymax = np.squeeze(xs[6])
        ymin = np.squeeze(xs[7])

        xnorm = mapminmax(x_nondim, xmin, xmax, ymin, ymax, direction='Forward')

    else:
        # Normalize with hardcoded process settings from training data
        xnorm = mapminmax(x_nondim, XMIN_INPUT, XMAX_INPUT, YMIN, YMAX, direction='Forward')

    # Load network weights from onnx file
    onnx_model = onnx.load(net_file)
    onnx.checker.check_model(onnx_model)

    # RUN NETWORK FORWARD PASS
    ort_sess = ort.InferenceSession(net_file)
    input_name = ort_sess.get_inputs()[0].name

    # Need to pass inputs in as shape (Nd, Ni)
    outputs = ort_sess.run(None, {input_name: xnorm.T.astype(np.float32)})
    ynorm = outputs[0].T  # (No, Nd)

    # REVERSE NORMALIZATION
    if data_file:
        # Load normalization settings from .mat file if specified
        data = scipy.io.loadmat(data_file)

        # Parse normalization settings from Matlab "process settings" struct
        ys = data['ys'][0][0]
        xmax = ys[2]
        xmin = ys[3]
        ymax = np.squeeze(ys[6])
        ymin = np.squeeze(ys[7])

        y = mapminmax(ynorm, xmin, xmax, ymin, ymax, direction='Reverse')

    else:
        # Un-Normalize with hardcoded process settings from training data
        y = mapminmax(ynorm, XMIN_OUTPUT, XMAX_OUTPUT, YMIN, YMAX, direction='Reverse')

    y = np.power(10, y)  # undo the logarithm from training

    # Get Bias voltage
    if V0 != V0_DEFAULT:
        V0 = np.atleast_1d(np.squeeze(V0))
        assert len(V0.shape) == 1
        if V0.shape[0] > 1:
            assert V0.shape[0] == y.shape[1]  # Must be same size as number of samples

    # RE-DIMENSIONALIZE
    E_max = np.multiply(y, np.divide(V0, h))

    return E_max.flatten()


def main():
    # EXAMPLE USAGE
    data_dir = Path('../data/base')
    test_data = np.loadtxt(str(data_dir/'test'/'samples'/'samples.txt'), skiprows=1)
    test_data[:, 2] = test_data[:, 2] * (np.pi / 180)

    E_max = forward(test_data.T, net_file=str(data_dir/'models'/'esi_surrogate.onnx'),
                    data_file=str(data_dir/'models'/'norm_data.mat'));
    plt.hist(E_max, density=True, bins=50, edgecolor='black')
    plt.show()


if __name__ == '__main__':
    main()
