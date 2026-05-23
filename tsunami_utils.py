import numpy as np
import numba


### MacCORMACK SOLVER ###
@numba.njit(parallel=True, cache=False)
def step_maccormack(eta, u, v, H, dx, dy, dt, g, coast_id):
    """
    One MacCormack predictor-corrector step of the 2D linear shallow-water equations

    coast_id:  0 = top (north)   reflective wall, v=0 at j=ny-1
               3 = right (east)  reflective wall, u=0 at i=nx-1
    """
    nx, ny = eta.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    # PREDICTOR (forward differences) 
    eta_p = np.zeros_like(eta)
    u_p   = np.zeros_like(u)
    v_p   = np.zeros_like(v)

    for i in numba.prange(0, nx - 1):
        for j in range(0, ny - 1):
            u_p[i, j] = u[i, j] - g * dt * (eta[i+1, j] - eta[i, j]) * inv_dx
            v_p[i, j] = v[i, j] - g * dt * (eta[i, j+1] - eta[i, j]) * inv_dy
            eta_p[i, j] = eta[i, j] - dt * (
                (H[i+1, j] * u[i+1, j] - H[i, j] * u[i, j]) * inv_dx +
                (H[i, j+1] * v[i, j+1] - H[i, j] * v[i, j]) * inv_dy
            )

    # predictor BCs: zero-gradient everywhere
    for j in numba.prange(ny):
        eta_p[0, j]    = eta_p[1, j];    u_p[0, j]    = u_p[1, j];    v_p[0, j]    = v_p[1, j]
        eta_p[nx-1, j] = eta_p[nx-2, j]; u_p[nx-1, j] = u_p[nx-2, j]; v_p[nx-1, j] = v_p[nx-2, j]
    for i in numba.prange(nx):
        eta_p[i, 0]    = eta_p[i, 1];    u_p[i, 0]    = u_p[i, 1];    v_p[i, 0]    = v_p[i, 1]
        eta_p[i, ny-1] = eta_p[i, ny-2]; u_p[i, ny-1] = u_p[i, ny-2]; v_p[i, ny-1] = v_p[i, ny-1]

    # reflective wall BC on predictor
    if coast_id == 0:   # top
        for i in numba.prange(nx): v_p[i, ny-1] = 0.0
    elif coast_id == 3: # right
        for j in numba.prange(ny): u_p[nx-1, j] = 0.0

    # CORRECTOR (backward differences on predicted state) 
    eta_new = np.empty_like(eta)
    u_new   = np.empty_like(u)
    v_new   = np.empty_like(v)

    for i in numba.prange(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = 0.5 * (
                u[i, j] + u_p[i, j]
                - g * dt * (eta_p[i, j] - eta_p[i-1, j]) * inv_dx
            )
            v_new[i, j] = 0.5 * (
                v[i, j] + v_p[i, j]
                - g * dt * (eta_p[i, j] - eta_p[i, j-1]) * inv_dy
            )
            eta_new[i, j] = 0.5 * (
                eta[i, j] + eta_p[i, j] - dt * (
                    (H[i, j] * u_p[i, j] - H[i-1, j] * u_p[i-1, j]) * inv_dx +
                    (H[i, j] * v_p[i, j] - H[i, j-1] * v_p[i, j-1]) * inv_dy
                )
            )

    # corrector BCs: zero-gradient everywhere
    for j in numba.prange(ny):
        eta_new[0, j]    = eta_new[1, j];    u_new[0, j]    = u_new[1, j];    v_new[0, j]    = v_new[1, j]
        eta_new[nx-1, j] = eta_new[nx-2, j]; u_new[nx-1, j] = u_new[nx-2, j]; v_new[nx-1, j] = v_new[nx-2, j]
    for i in numba.prange(nx):
        eta_new[i, 0]    = eta_new[i, 1];    u_new[i, 0]    = u_new[i, 1];    v_new[i, 0]    = v_new[i, 1]
        eta_new[i, ny-1] = eta_new[i, ny-2]; u_new[i, ny-1] = u_new[i, ny-2]; v_new[i, ny-1] = v_new[i, ny-1]

    # reflective wall BC on corrector
    if coast_id == 0:   # top
        for i in numba.prange(nx): v_new[i, ny-1] = 0.0
    elif coast_id == 3: # right
        for j in numba.prange(ny): u_new[nx-1, j] = 0.0

    return eta_new, u_new, v_new


@numba.njit(parallel=True, cache=False)
def apply_sponge(eta, u, v, mask):
    """ Multiply all fields by the sponge mask to damp open boundaries """
    nx, ny = eta.shape
    for i in numba.prange(nx):
        for j in range(ny):
            eta[i, j] *= mask[i, j]
            u[i, j]   *= mask[i, j]
            v[i, j]   *= mask[i, j]
    return eta, u, v


### SPONGE MASK ###
def make_sponge(nx, ny, coast, width=20):
    """
    Cosine-taper sponge mask. Values ramp from 0 (boundary) to 1 (interior)
    on all four edges except the reflective coast, which stays at 1.
    coast = "top"   : north wall is the coast (simple model)
    coast = "right" : east wall is the coast  (Mavericks model)
    """
    mask = np.ones((nx, ny))
    for k in range(width):
        fac = 0.5 * (1.0 - np.cos(np.pi * k / width))
        mask[k, :]      = np.minimum(mask[k, :],      fac)   # left
        mask[nx-1-k, :] = np.minimum(mask[nx-1-k, :], fac)   # right
        mask[:, k]      = np.minimum(mask[:, k],       fac)   # bottom
        mask[:, ny-1-k] = np.minimum(mask[:, ny-1-k],  fac)   # top

    # restore the reflective coast side
    if coast == "top":
        mask[:, ny-width:] = 1.0
    elif coast == "right":
        mask[nx-width:, :] = 1.0

    return mask


### HELPERS ###
def smooth_1d(arr, size=7):
    """
    Box-car smoothing along a 1D array.
    Used to smooth the peak coastal amplitude profile, which is noisy
    because the jagged GEBCO coastline causes coast_i to jump between
    adjacent grid cells. Smoothing reveals the underlying signal
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(arr, size=size)
