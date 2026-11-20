"""Vectorised negative-binomial GLM utilities.

The model is log(mu) = X beta + log(size_factor).  Dispersions follow a
DESeq2-style workflow at gene level: grid maximum likelihood, a binned
median trend over expression level, and empirical-Bayes shrinkage toward
that trend.  All routines operate on (n_samples x n_genes) matrices and
loop only over gene chunks.
"""

import numpy as np
from scipy import optimize, special, stats


def factorial_matrix(a, d, b, interactions=True):
    """Design matrix for the 2x2x2 factorial with A, D, B coded 0/1."""
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    b = np.asarray(b, dtype=float)
    cols = [np.ones_like(a), a, d, b]
    names = ["intercept", "A", "D", "B"]
    if interactions:
        cols += [a * d, a * b, d * b, a * d * b]
        names += ["A:D", "A:B", "D:B", "A:D:B"]
    return np.column_stack(cols), names


def poscounts_size_factors(counts):
    """Size factors following DESeq2's 'poscounts' estimator.

    log counts of zero entries are set to 0 before averaging, and each
    sample's factor is the median of log ratios over genes with a
    positive count in that sample.
    """
    counts = np.asarray(counts, dtype=np.float64)
    logc = np.where(counts > 0, np.log(np.maximum(counts, 1.0)), 0.0)
    log_geo = logc.mean(axis=0)
    all_zero = counts.sum(axis=0) == 0
    logratios = np.where(
        (counts > 0) & (~all_zero)[None, :],
        np.log(np.maximum(counts, 1.0)) - log_geo[None, :],
        np.nan,
    )
    sf = np.exp(np.nanmedian(logratios, axis=1))
    return sf / np.exp(np.mean(np.log(sf)))


def median_of_ratios(counts):
    """Size factors from the median-of-ratios estimator (DESeq2)."""
    counts = np.asarray(counts, dtype=np.float64)
    positive = (counts > 0).all(axis=0)
    if positive.sum() < 100:
        raise ValueError("too few genes with positive counts in all samples")
    logc = np.log(counts[:, positive])
    log_geo = logc.mean(axis=0)
    sf = np.exp(np.median(logc - log_geo, axis=1))
    return sf / np.exp(np.mean(np.log(sf)))


def nb_loglik(y, mu, alpha):
    """NB2 log likelihood per gene (shape alpha), summed over samples."""
    a = np.asarray(alpha, dtype=np.float64)[None, :]
    r = 1.0 / a
    return (
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1.0)
        + r * np.log(r / (r + mu))
        + y * np.log(mu / (r + mu))
    ).sum(axis=0)


def nb_deviance(y, mu, alpha):
    a = np.asarray(alpha, dtype=np.float64)[None, :]
    t1 = np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu), 0.0)
    t2 = (y + 1.0 / a) * np.log((1.0 + a * y) / (1.0 + a * mu))
    return 2.0 * (t1 - t2).sum(axis=0)


def irls(y, x, offset, alpha, beta=None, max_iter=40, tol=1e-7):
    """Fit NB GLM per gene by IRLS with fixed dispersion."""
    n, p = x.shape
    alpha = np.asarray(alpha, dtype=np.float64)
    if beta is None:
        mu0 = np.maximum(y, 0.5)
        beta = np.linalg.lstsq(x, np.log(mu0) - offset[:, None], rcond=None)[0]
    else:
        beta = beta.copy()
    dev_prev = None
    iterations = 0
    for it in range(max_iter):
        eta = np.clip(x @ beta + offset[:, None], -30.0, 30.0)
        mu = np.exp(eta)
        w = mu / (1.0 + alpha[None, :] * mu)
        z = eta - offset[:, None] + (y - mu) / mu
        xtwx = np.einsum("np,ng,nq->gpq", x, w, x)
        xtwx += np.eye(p)[None, :, :] * 1e-9
        xtwz = np.einsum("np,ng->gp", x, w * z)
        beta = np.linalg.solve(xtwx, xtwz[..., None])[..., 0].T
        iterations = it + 1
        dev = nb_deviance(y, mu, alpha)
        if dev_prev is not None and np.all(
            np.abs(dev - dev_prev) <= tol * (np.abs(dev_prev) + 1.0)
        ):
            break
        dev_prev = dev
    eta = np.clip(x @ beta + offset[:, None], -30.0, 30.0)
    return {"beta": beta, "mu": np.exp(eta), "iterations": iterations}


def nb_cov(x, mu, alpha):
    """Per-gene coefficient covariance matrices from the final IRLS weights."""
    w = mu / (1.0 + np.asarray(alpha)[None, :] * mu)
    xtwx = np.einsum("np,ng,nq->gpq", x, w, x)
    xtwx += np.eye(x.shape[1])[None, :, :] * 1e-9
    return np.linalg.inv(xtwx)


def cox_reid_adjustment(x, mu, alpha):
    """Cox-Reid adjustment 0.5 * log|X' W X| for dispersion estimation."""
    w = mu / (1.0 + np.asarray(alpha)[None, :] * mu)
    xtwx = np.einsum("np,ng,nq->gpq", x, w, x)
    xtwx += np.eye(x.shape[1])[None, :, :] * 1e-9
    _, logdet = np.linalg.slogdet(xtwx)
    return 0.5 * logdet


def estimate_dispersions(y, x, offset, n_grid=12, lo=1e-4, hi=10.0):
    """Gene-wise dispersion by Cox-Reid adjusted grid likelihood."""
    n_genes = y.shape[1]
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    loglik = np.empty((n_grid, n_genes))
    beta = None
    for k, a in enumerate(grid):
        fit = irls(y, x, offset, np.full(n_genes, a), beta=beta, max_iter=25, tol=1e-7)
        beta = fit["beta"]
        loglik[k] = (
            nb_loglik(y, fit["mu"], np.full(n_genes, a))
            - cox_reid_adjustment(x, fit["mu"], np.full(n_genes, a))
        )

    best = loglik.argmax(axis=0)
    step = np.log(grid[1] / grid[0])
    log_alpha = np.log(grid[best])
    interior = (best > 0) & (best < n_grid - 1)
    if interior.any():
        idx = np.arange(n_genes)[interior]
        y0 = loglik[best[idx] - 1, idx]
        y1 = loglik[best[idx], idx]
        y2 = loglik[best[idx] + 1, idx]
        denom = y0 - 2.0 * y1 + y2
        curv = np.where(np.abs(denom) > 1e-12, denom, np.nan)
        delta = np.clip(0.5 * (y0 - y2) / curv, -1.0, 1.0)
        delta = np.where(np.isfinite(delta), delta, 0.0)
        log_alpha[idx] = log_alpha[idx] + delta * step
    alpha = np.clip(np.exp(log_alpha), 1e-6, 50.0)
    return alpha, loglik.max(axis=0)


def dispersion_trend(base_mean, alpha_gene, n_bins=30):
    """Binned local-median trend of dispersion over mean expression."""
    ok = np.isfinite(alpha_gene) & (alpha_gene > 0) & (base_mean > 0)
    x = np.log10(base_mean[ok])
    ylog = np.log(alpha_gene[ok])
    order = np.argsort(x)
    x, ylog = x[order], ylog[order]
    bins = np.array_split(np.arange(len(x)), n_bins)
    cx = np.array([np.median(x[b]) for b in bins])
    cy = np.array([np.median(ylog[b]) for b in bins])
    padded = np.r_[cy[0], cy, cy[-1]]
    cy = np.convolve(padded, np.ones(3) / 3.0, mode="valid")
    trend = np.exp(
        np.interp(np.log10(np.clip(base_mean, 1e-8, None)), cx, cy,
                  left=cy[0], right=cy[-1])
    )
    return trend


def shrink_dispersions(alpha_gene, alpha_trend, df_resid, safety=1.2):
    """Empirical-Bayes shrinkage toward the trend (DESeq2-style).

    The constant ``safety`` is a dispersion inflation factor chosen by
    null-model simulation so that the LRT keeps close to nominal size
    when dispersions are estimated from three replicates per group.
    """
    resid = np.log(alpha_gene) - np.log(alpha_trend)
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    sampling_var = special.polygamma(1, df_resid / 2.0)
    var_prior = max(mad ** 2 - sampling_var, 0.05)
    f = lambda d: special.polygamma(1, d / 2.0) - var_prior
    try:
        df_prior = optimize.brentq(f, 0.2, 500.0)
    except ValueError:
        df_prior = 100.0
    weight = df_resid / (df_resid + df_prior)
    alpha = weight * alpha_gene + (1.0 - weight) * alpha_trend
    return np.clip(alpha * safety, 1e-6, 50.0), df_prior


def lrt(y, x_full, x_red, offset, alpha, beta_full=None):
    """Likelihood-ratio test of the full model against a reduced model."""
    fit_full = irls(y, x_full, offset, alpha, beta=beta_full)
    fit_red = irls(y, x_red, offset, alpha)
    ll_full = nb_loglik(y, fit_full["mu"], alpha)
    ll_red = nb_loglik(y, fit_red["mu"], alpha)
    stat = np.maximum(2.0 * (ll_full - ll_red), 0.0)
    df = x_full.shape[1] - x_red.shape[1]
    pval = stats.chi2.sf(stat, df=df)
    return {
        "stat": stat,
        "pvalue": pval,
        "ll_full": ll_full,
        "ll_red": ll_red,
        "fit_full": fit_full,
        "fit_red": fit_red,
    }


def wald_contrast(beta, cov, contrast):
    """Wald test of one linear combination of coefficients per gene."""
    c = np.asarray(contrast, dtype=np.float64)
    est = c @ beta
    var = np.einsum("p,gpq,q->g", c, cov, c)
    se = np.sqrt(np.maximum(var, 1e-300))
    z = est / se
    pval = 2.0 * stats.norm.sf(np.abs(z))
    return {"estimate": est, "se": se, "z": z, "pvalue": pval}
