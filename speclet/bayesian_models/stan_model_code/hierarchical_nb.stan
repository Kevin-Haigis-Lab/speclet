// Hierarchical negative binomial model.

data {
    int<lower=0> N;  // number of data points
    int<lower=1> S;  // number of sgRNAs
    int<lower=1> G;  // number of genes
    int<lower=1> C;  // number of cell lines
    array[N] real<lower=0> ct_initial;
    array[N] int<lower=0> ct_final;
    array[N] int<lower=1, upper=S> sgrna_idx;
    array[N] int<lower=1, upper=G> gene_idx;
    array[S] int<lower=1, upper=G> sgrna_to_gene_idx;
    array[N] int<lower=1, upper=C> cellline_idx;
}

parameters {
    real mu_mu_beta;
    real<lower=0> sigma_mu_beta;
    array[G] real mu_beta;
    real<lower=0> sigma_beta;
    array[S] real beta_s;
    real<lower=0> sigma_gamma;
    array[C] real delta_gamma;
    real<lower=0> sigma_kappa;
    array[S,C] real delta_kappa;
    real<lower=0> alpha_alpha;
    real<lower=0> beta_alpha;
    array[G] real<lower=0> alpha;
}

transformed parameters {
    array[N] real eta;
    array[N] real<lower=0> mu;
    array[S,C] real kappa_sc;
    array[C] real gamma_c;

    for (c in 1:C) {
        gamma_c[c] = 0.0 + delta_gamma[c] * sigma_gamma;
    }

    for (s in 1:S) {
        for (c in 1:C) {
            kappa_sc[s,c] = 0.0 + delta_kappa[s,c] * sigma_kappa;
        }
    }

    for (n in 1:N) {
        eta[n] = beta_s[sgrna_idx[n]] + gamma_c[cellline_idx[n]] + kappa_sc[sgrna_idx[n]][cellline_idx[n]];
        mu[n] = exp(eta[n]) * ct_initial[n];
    }
}

model {
    // Priors
    mu_mu_beta ~ normal(0.0, 5.0);
    sigma_mu_beta ~ gamma(2.0, 0.5);
    sigma_beta ~ gamma(2.0, 0.5);
    sigma_gamma ~ gamma(2.0, 0.5);
    sigma_kappa ~ gamma(1.1, 0.5);
    alpha_alpha ~ gamma(2.0, 0.5);
    beta_alpha ~ gamma(2.0, 0.5);
    mu_beta ~ normal(mu_mu_beta, sigma_mu_beta);
    delta_gamma ~ normal(0.0, 1.0);
    beta_s ~ normal(mu_beta[sgrna_to_gene_idx], sigma_beta);

    for (s in 1:S) {
        delta_kappa[s] ~ normal(0.0, 1.0);
    }

    alpha ~ gamma(alpha_alpha, beta_alpha);
    ct_final ~ neg_binomial_2(mu, alpha[gene_idx]);
}

generated quantities {
    vector[N] log_lik;
    array[N] real<lower=0> y_hat;

    y_hat = neg_binomial_2_rng(mu, alpha[gene_idx]);
    for (n in 1:N) {
        log_lik[n] = neg_binomial_2_lpmf(ct_final[n] | mu[n], alpha[gene_idx[n]]);
    }
}
