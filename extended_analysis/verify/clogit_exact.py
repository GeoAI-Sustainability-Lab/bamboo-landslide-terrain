# -*- coding: utf-8 -*-
"""Exact conditional logistic regression (vectorised, log-space).

The conditional likelihood denominator for a stratum is the elementary symmetric
polynomial e_m(exp(eta_1),...,exp(eta_n)), computed by a dynamic program that is
vectorised over the subset-size index, so no recursion and no combinatorial blow-up.
Validated against statsmodels ConditionalLogit.
"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import numpy as np
from scipy.optimize import minimize

NEG = -np.inf

def _stratum(eta, xs, m):
    """returns (log e_m, d log e_m / d beta) for one stratum"""
    n, p = xs.shape
    dp = np.full(m + 1, NEG); dp[0] = 0.0
    dg = np.zeros((m + 1, p))
    for j in range(n):
        e = eta[j]; xj = xs[j]
        hi = min(m, j + 1)
        a = dp[1:hi + 1]                       # old dp[k]
        b = dp[0:hi] + e                       # old dp[k-1] + eta_j
        mx = np.maximum(a, b)
        fin = np.isfinite(mx)
        wa = np.where(fin & np.isfinite(a), np.exp(np.where(fin, a - mx, 0.0)), 0.0)
        wb = np.where(fin & np.isfinite(b), np.exp(np.where(fin, b - mx, 0.0)), 0.0)
        tot = wa + wb
        newdp = np.where(tot > 0, mx + np.log(np.where(tot > 0, tot, 1.0)), NEG)
        newdg = np.where(tot[:, None] > 0,
                         (wa[:, None] * dg[1:hi + 1] + wb[:, None] * (dg[0:hi] + xj)) /
                         np.where(tot[:, None] > 0, tot[:, None], 1.0), 0.0)
        dp[1:hi + 1] = newdp
        dg[1:hi + 1] = newdg
    return dp[m], dg[m]

def _obj(beta, groups, X, y):
    nll = 0.0; grad = np.zeros(len(beta))
    for idx in groups:
        xs = X[idx]; ys = y[idx]
        eta = xs @ beta
        m = int(ys.sum())
        ld, dgm = _stratum(eta, xs, m)
        nll += ld - float(eta[ys == 1].sum())
        grad += dgm - xs[ys == 1].sum(axis=0)
    return nll, grad

def fit(y, X, groups_labels, maxiter=300, hstep=1e-5):
    y = np.asarray(y, float); X = np.asarray(X, float)
    lab = np.asarray(groups_labels)
    order = np.argsort(lab, kind='stable'); lab_s = lab[order]
    bnd = np.flatnonzero(np.r_[True, lab_s[1:] != lab_s[:-1], True])
    groups = [order[bnd[i]:bnd[i + 1]] for i in range(len(bnd) - 1)]
    groups = [g for g in groups if 0 < y[g].sum() < len(g)]
    f = lambda b: _obj(b, groups, X, y)
    r = minimize(f, np.zeros(X.shape[1]), jac=True, method='BFGS',
                 options=dict(maxiter=maxiter, gtol=1e-7))
    p = len(r.x); H = np.zeros((p, p))
    for i in range(p):
        e = np.zeros(p); e[i] = hstep
        _, g1 = f(r.x + e); _, g0 = f(r.x - e)
        H[:, i] = (g1 - g0) / (2 * hstep)
    H = 0.5 * (H + H.T)
    return dict(params=r.x, bse=np.sqrt(np.diag(np.linalg.inv(H))), nll=float(r.fun),
                n_strata=len(groups), converged=bool(r.success))
