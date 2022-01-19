// Hierarchical negative binomial model.

data {
    int<lower=1> N;  // number of data points
    int<lower=1> S;  // number of sgRNAs
    int<lower=1> G;  // number of genes
    real<lower=0> ct_initial[N];
    int<lower=0> ct_final[N];
    int<lower=1> sgrna_idx[N];
    int<lower=1> sgrna_to_gene_idx[S];
}

parameters {
    real mu_mu_beta;
    real<lower=0> sigma_mu_beta;
    real mu_beta[G];
    real<lower=0> sigma_beta;
    real beta_s[S];
    real<lower=0> alpha;
}

transformed parameters {
    real mu[N];

    for (i in 1:N) {
        mu[i] = exp(beta_s[sgrna_idx[i]]) * ct_initial[i];
    }
}

model {
    // Priors
    mu_mu_beta ~ normal(0, 5);
    sigma_mu_beta ~ gamma(2.0, 0.5);
    sigma_beta ~ gamma(2.0, 0.5);
    alpha ~ gamma(2.0, 0.2);

    for (g in 1:G) {
        mu_beta[g] ~ normal(mu_mu_beta, sigma_mu_beta);
    }

    for (s in 1:S) {
        beta_s[s] ~ normal(mu_beta[sgrna_to_gene_idx[s]], sigma_beta);
    }

    ct_final ~ neg_binomial_2(mu, alpha);
}

generated quantities {
    vector[N] log_lik;
    vector[N] y_hat;

    for (i in 1:N) {
        log_lik[i] = neg_binomial_2_lpmf(ct_final[i] | mu[i], alpha);
        y_hat[i] = neg_binomial_2_rng(mu[i], alpha);
    }
}
