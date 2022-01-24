// Hierarchical negative binomial model.

data {
    int<lower=1> N;  // number of data points
    int<lower=1> S;  // number of sgRNAs
    int<lower=1> G;  // number of genes
    int<lower=1> C;  // number of cell lines
    real<lower=0> ct_initial[N];
    int<lower=0> ct_final[N];
    int<lower=1> sgrna_idx[N];
    int<lower=1> gene_idx[N];
    int<lower=1> sgrna_to_gene_idx[S];
    int<lower=1> cellline_idx[N];
}

parameters {
    real mu_mu_beta;
    real<lower=0> sigma_mu_beta;
    real mu_beta[G];
    real<lower=0> sigma_beta;
    real beta_s[S];
    real<lower=0> sigma_gamma;
    real delta_gamma[C];
    real<lower=0> sigma_kappa;
    vector[C] delta_kappa[S];
    real<lower=0> alpha_alpha;
    real<lower=0> beta_alpha;
    real<lower=0> alpha[G];
}

transformed parameters {
    real eta[N];
    real mu[N];
    real kappa_sc[S,C];
    real gamma_c[C];

    for (c in 1:C) {
        gamma_c[c] = 0.0 + delta_gamma[c] * sigma_gamma;
    }

    for (s in 1:S) {
        for (c in 1:C) {
            kappa_sc[s][c] = 0.0 + delta_kappa[s][c] * sigma_kappa;
        }
    }

    for (i in 1:N) {
        eta[i] = beta_s[sgrna_idx[i]] + gamma_c[cellline_idx[i]] + kappa_sc[sgrna_idx[i], cellline_idx[i]];
        mu[i] = exp(eta[i]) * ct_initial[i];
    }
}

model {
    // Priors
    mu_mu_beta ~ normal(0.0, 5.0);
    sigma_mu_beta ~ gamma(2.0, 0.5);
    sigma_beta ~ gamma(2.0, 0.5);
    sigma_gamma ~ gamma(2.0, 0.5);
    sigma_kappa ~ gamma(1.1, 0.5);
    alpha_alpha ~ normal(0.0, 2.5);
    beta_alpha ~ normal(0.0, 1.0);

    mu_beta ~ normal(mu_mu_beta, sigma_mu_beta);
    delta_gamma ~ normal(0.0, 1.0);

    for (s in 1:S) {
        beta_s[s] ~ normal(mu_beta[sgrna_to_gene_idx[s]], sigma_beta);
        delta_kappa[s] ~ normal(0.0, 1.0);
    }

    alpha ~ gamma(alpha_alpha, beta_alpha);

    for (i in 1:N) {
        ct_final ~ neg_binomial_2(mu[i], alpha[gene_idx[i]]);
    }
}

generated quantities {
    vector[N] log_lik;
    vector[N] y_hat;

    for (i in 1:N) {
        log_lik[i] = neg_binomial_2_lpmf(ct_final[i] | mu[i], alpha[gene_idx[i]]);
        y_hat[i] = neg_binomial_2_rng(mu[i], alpha[gene_idx[i]]);
    }
}
