import numpy as np

import viz

def diffusion_step(Xmid, t, get_mu_sigma, denoise_sigma, mask, XT, rng):
    """
    Run a single reverse diffusion step
    """
    mu, sigma = get_mu_sigma(Xmid, np.array([[t]]))
    if denoise_sigma is not None:
        sigma_new = (sigma**-2 + denoise_sigma**-2)**-0.5
        mu_new = mu * sigma_new**2 * sigma**-2 + XT * sigma_new**2 * denoise_sigma**-2
        sigma = sigma_new
        mu = mu_new
    if mask is not None:
        mu.flat[mask] = XT.flat[mask]
        sigma.flat[mask] = 0.
    Xmid = mu + sigma*rng.normal(size=Xmid.shape)
    return Xmid


def generate_inpaint_mask(n_samples, n_colors, spatial_width):
    """
    The mask will be True where we keep the true image, and False where we're
    inpainting.
    """
    mask = np.zeros((n_samples, n_colors, spatial_width, spatial_width), dtype=bool)
    # simple mask -- just mask out half the image   
    mask[:,:,:,spatial_width/2:] = True
    return mask.ravel()


def generate_samples(model, get_mu_sigma,
            n_samples=36, inpaint=False, denoise_sigma=None, X_true=None,
            base_fname_part1="samples", base_fname_part2='',
            num_intermediate_plots=4, seed=12345):
    """
    Run the reverse diffusion process (generative model).
    """
    # use the same noise in the samples every time, so they're easier to
    # compare across learning
    rng = np.random.RandomState(seed)

    spatial_width = model.spatial_width
    n_colors = model.n_colors
    trajectory_length = model.trajectory_length

    # set the initial state X^T of the reverse trajectory
    XT = rng.normal(size=(n_samples,n_colors,spatial_width,spatial_width))
    if denoise_sigma is not None:
        XT = X_true + XT*denoise_sigma
        base_fname_part1 += '_denoise%g'%denoise_sigma
    if inpaint:
        mask = generate_inpaint_mask(n_samples, n_colors, spatial_width)
        XT.flat[mask] = X_true.flat[mask]
        base_fname_part1 += '_inpaint'
    else:
        mask = None

    if X_true is not None:
        viz.plot_images(X_true, base_fname_part1 + '_true' + base_fname_part2)
    viz.plot_images(XT, base_fname_part1 + '_t%04d'%model.trajectory_length + base_fname_part2)

    Xmid = XT.copy()
    for t in xrange(model.trajectory_length-1, 0, -1):
        Xmid = diffusion_step(Xmid, t, get_mu_sigma, denoise_sigma, mask, XT, rng)
        if np.mod(model.trajectory_length-t,
            int(np.ceil(model.trajectory_length/(num_intermediate_plots+2.)))) == 0:
            viz.plot_images(Xmid, base_fname_part1 + '_t%04d'%t + base_fname_part2)

    X0 = Xmid
    viz.plot_images(X0, base_fname_part1 + '_t%04d'%0 + base_fname_part2)
