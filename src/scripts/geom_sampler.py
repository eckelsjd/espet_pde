from pathlib import Path
import numpy as np
import random

# Bounds on geometry parameters
H_BOUNDS = [50, 1000]       # um
ALPHA_BOUNDS = [10, 70]     # deg
RC_BOUNDS = [1, 100]        # um
D_BOUNDS = [-1000, 3000]    # um
RA_BOUNDS = [10, 3000]      # um


def sampler(n, max_tries=100):
    i = 0
    try_count = 0
    samples = np.zeros((n, 5))
    while i < n:
        h = random.uniform(H_BOUNDS[0], H_BOUNDS[1])
        alpha = random.uniform(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])
        rc = random.uniform(RC_BOUNDS[0], RC_BOUNDS[1])
        d = random.uniform(D_BOUNDS[0], D_BOUNDS[1])
        ra = random.uniform(RA_BOUNDS[0], RA_BOUNDS[1])

        if check_geom_constraints(d, rc, alpha, h, ra):
            samples[i, :] = [d*1e-6, rc*1e-6, alpha, h*1e-6, ra*1e-6]
            i += 1

            if try_count > 0:
                try_count = 0
                print(f'RESET for i={i}')
        else:
            try_count += 1
            print(f'RETRY {try_count} OUT OF {max_tries} for i={i}')
            if try_count >= max_tries:
                print(f'MAX RETRIES LIMIT HIT. SAMPLING FAILED')
                break

    return samples


def check_base_constraints(d, rc, alpha, h, ra):
    RP_BOUNDS = [1e-6, 2e-5]
    rp = random.uniform(RP_BOUNDS[0], RP_BOUNDS[1])

    # Radius of curvature lower bound
    if rc <= 10*rp:
        return False

    # Cone half-angle upper bound
    P = 0.48
    Npores = 10
    right = 1.0 - (Npores / (3 * P)) * (rp / rc) ** 2
    alpha_ub = (180 / np.pi) * np.arcsin(right)
    if alpha >= alpha_ub:
        return False

    # Emitter height lower bound
    if h <= rc * (1 - np.sin(alpha * (np.pi / 180))):
        return False

    # Aperture radius bounds
    alpha_rad = alpha * (np.pi / 180)
    rb = rc * np.sin((np.pi / 2) - alpha_rad) + (h - rc * (1 - np.cos((np.pi / 2) - alpha_rad))) * np.tan(alpha_rad)
    if ra <= d * np.tan(25*np.pi/180) or ra >= rb:
        return False

    return True


def check_geom_constraints(d, rc, alpha, h, ra):
    alpha_rad = alpha * (np.pi/180)

    # Check lower height bound (5% for numerical safety)
    hl = rc*(1-np.sin(alpha_rad))
    if h < 1.05*hl:
        return False

    # Check non-contact constraint if d < 0
    if d < 0:
        # Cone base radius
        rb = rc*np.cos(alpha_rad) + (h-hl)*np.tan(alpha_rad)

        # Cone height
        h0 = rb/np.tan(alpha_rad)

        # Get emitter cross-sectional radius at r(y=y_crit)
        y_crit = h - np.abs(d)
        if 0 < y_crit < h-hl:
            r_crit = rb - (rb/h0)*y_crit
        elif h-hl <= y_crit <= h:
            yc = h-rc
            r_crit = np.sqrt(rc**2 - (y_crit - yc)**2)
        else:
            return False  # Badly-formed geometry

        # 5% margin for numerical safety
        if r_crit >= ra*0.95:
            return False

    return True


if __name__ == '__main__':
    # d = 360e-6
    # rc = 1.6e-5
    # alpha = 30
    # h = 350e-6
    # ras = np.linspace(1e-5, 4e-3, 128)
    # check_pcts = np.zeros(ras.shape)
    # for i, ra in enumerate(ras):
    #     count = 0
    #     for j in range(1000):
    #         if check_base_constraints(d, rc, alpha, h, ra):
    #             count += 1
    #
    #     check_pcts[i] = count/100
    #     print(f'i={i} pct={check_pcts[i]*100:.02f}')

    # d = -1.463e-04
    # rc = 6.7279e-5
    # alpha = 37.73
    # h = 7.9799e-4
    # ra = 3.9711e-5
    # b = check_geom_constraints(d*1e6, rc*1e6, alpha, h*1e6, ra*1e6)

    sdir = Path('../../data/geometry/samples')
    sfile = sdir / 'samples.txt'
    fd = open(sfile, 'w')
    fd.write('d rc alpha h ra\n')

    n = 200
    samples = sampler(n)

    for i in range(n):
        params = samples[i, :]
        wstring = f"{params[0]} {params[1]} {params[2]} {params[3]} {params[4]}\n"
        fd.write(wstring)

    fd.close()
