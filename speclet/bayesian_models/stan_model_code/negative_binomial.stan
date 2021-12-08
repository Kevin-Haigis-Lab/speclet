// Negative binomial generalized linear model with a single covariate.

data {
  int<lower=0> N;
  real<lower=0> ct_initial[N];
  real<lower=0> ct_final[N];
}

parameters {
    real alpha;
    real beta;
    real<lower=0> reciprocal_phi;
}

transformed parameters {
    vector[N] eta;
    real phi;
    eta = beta;
    phi = 1.0 / reciprocal_phi;
}

model {
    reciprocal_phi ~ cauchy(0.0, 10.0);
    beta ~ normal(0.0, 5.0);
    ct_final ~ neg_binomial_2(eta * ct_initial, phi);
}
