// Negative binomial generalized linear model with a single covariate.

data {
  int<lower=0> N;
  real<lower=0> ct_initial[N];
  int<lower=0> ct_final[N];
}

parameters {
    real beta;
    real<lower=0> reciprocal_phi;
}

transformed parameters {
    real eta;
    real mu[N];
    real phi;

    eta = beta;
    for (i in 1:N) {
        mu[i] = exp(eta) * ct_initial[i];
    }

    phi = 1.0 / reciprocal_phi;
}

model {
    // Priors
    reciprocal_phi ~ cauchy(0.0, 10.0);
    beta ~ normal(0.0, 5.0);

    // Likelihood
    ct_final ~ neg_binomial_2(mu, phi);
}
