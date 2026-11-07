"""A low-default corporate portfolio, end to end, on made-up numbers."""

import numpy as np

import eqlx_credit as ec

# Obligors and defaults per grade, lowest risk first (synthetic).
n_obligors = np.array([120, 340, 510, 430, 260, 90])
n_defaults = np.array([0, 0, 1, 2, 3, 2])

# 1. Too few defaults to trust raw default rates: prudent upper bounds instead.
pt = ec.pluto_tasche(n_obligors, n_defaults, confidence=0.9, rho=0.12)
print("Pluto-Tasche (90%, rho=0.12):", np.round(pt * 100, 3), "%")

# 2. A smooth PD curve from a long-run central tendency and a target accuracy ratio.
fit = ec.fit_ct_ar(n_obligors, central_tendency=0.015, accuracy_ratio=0.65)
print("CT/AR curve:                ", np.round(fit.pd * 100, 3), "%")

vdb = ec.van_der_burgt(n_obligors, central_tendency=0.015, accuracy_ratio=0.65)
print("van der Burgt:              ", np.round(vdb.pd * 100, 3), "%")

# 3. Put the grades on a 20-grade geometric master scale.
scale = ec.geometric_master_scale(0.0003, 0.30, 20)
print("Master scale grades:        ", scale.grade(fit.pd))

# 4. Back-test the fitted PDs against what happened.
jt = ec.jeffreys_test(n_obligors, n_defaults, fit.pd)
print("Jeffreys p-values:          ", np.round(jt.p_value, 3))
print("Grades flagged:             ", np.flatnonzero(jt.rejected) + 1)
print("Concentration (HHI):        ", round(ec.hhi(n_obligors), 3))
