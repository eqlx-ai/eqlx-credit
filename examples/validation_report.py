"""An annual PD validation pass on a grade table, end to end (synthetic numbers)."""

import numpy as np

import eqlx_credit as ec

# Grade table at the start of the year, lowest risk first (synthetic).
n = np.array([820, 1450, 1910, 1320, 640, 210])
d = np.array([1, 4, 11, 19, 21, 17])
pd = np.array([0.0012, 0.0035, 0.0070, 0.0140, 0.0300, 0.0700])

# 1. Discriminatory power, with a confidence interval, against the AUC at initial validation.
cur = ec.auc_ci_from_counts(n, d)
print(f"AUC {cur.auc:.3f} (95% CI {cur.ci_low:.3f}-{cur.ci_high:.3f}), Gini {cur.gini:.3f}")
drop = ec.auc_decline_test(0.80, cur)
print(f"Decline vs initial 0.800: S = {drop.statistic:.2f}, p = {drop.p_value:.3f}")

# 2. Calibration per grade: Jeffreys, and a traffic light that allows for correlated defaults.
rows = ec.jeffreys_test(n, d, pd).rows()
colours = ec.traffic_light(n, d, pd, rho=0.12).colours
for row, colour in zip(rows, colours, strict=True):
    print(
        f"grade {row['grade']}: {row['defaults']:>3} defaults vs {row['expected_defaults']:5.1f} expected, "
        f"Jeffreys p = {row['p_value']:.3f}, traffic light {colour}"
    )

# 3. Across years: the normal test on the portfolio default rate.
dr = np.array([0.0105, 0.0120, 0.0098, 0.0141, 0.0119])
print(f"Normal test over 5 years: z = {ec.normal_test(dr, np.full(5, 0.011)).statistic:.2f}")

# 4. Capital: Basel IRB risk weights for the grade PDs (corporate, LGD 45%, M 2.5).
rw = ec.risk_weight(pd, 0.45)
print("Risk weights:", " ".join(f"{x:.1%}" for x in rw))
