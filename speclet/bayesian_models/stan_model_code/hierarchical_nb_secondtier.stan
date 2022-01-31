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
    real mu_a;
    real<lower=0> sigma_a;
    array[G] real a_g;
    real<lower=0> sigma_b;
    array[C] real delta_b;
    real<lower=0> sigma_gamma;
    array[G,C] real delta_gamma;
    real<lower=0> sigma_beta;
    array[S,C] real delta_beta;
    real<lower=0> alpha_alpha;
    real<lower=0> beta_alpha;
    array[G] real<lower=0> alpha;
}

// Consider switching order of dimensions to have cell line as the first so can combine
// in for loops for 2D arrays. Would also want to swap in the PyMC3 version to keep
// consistent.

transformed parameters {
    array[G,C] real mu_beta;
    array[C] real b_c;
    array[G,C] real gamma_gc;
    array[S,C] real beta_sc;
    array[N] real eta;
    array[N] real mu;

    for (c in 1:C) {
        b_c[c] = 0.0 + delta_b[c] * sigma_b;
    }

    for (g in 1:G) {
        for (c in 1:C){
            gamma_gc[g,c] = 0.0 + delta_gamma[g,c] * sigma_gamma;
            mu_beta[g,c] = a_g[g] + b_c[c] + gamma_gc[g,c];
        }
    }

    for (s in 1:S) {
        for (c in 1:C) {
            beta_sc[s,c] = mu_beta[sgrna_to_gene_idx[s],c] + delta_beta[s,c] * sigma_beta;
        }
    }

    for (n in 1:N) {
        eta[n] = beta_sc[sgrna_idx[n], cellline_idx[n]];
        mu[n] = exp(eta[n]) * ct_initial[n];
    }
}

model {
    // Priors
    mu_a ~ normal(0.0, 5.0);
    sigma_a ~ gamma(2.0, 0.5);
    sigma_b ~ gamma(1.1, 0.5);
    sigma_gamma ~ gamma(1.1, 0.5);
    sigma_beta ~ gamma(2.0, 0.5);
    alpha_alpha ~ gamma(2.0, 0.5);
    beta_alpha ~ gamma(2.0, 0.5);

    a_g ~ normal(mu_a, sigma_a);
    delta_b ~ normal(0.0, 1.0);

    for (g in 1:G) {
        delta_gamma[g] ~ normal(0.0, 1.0);
    }
    for (s in 1:S) {
        delta_beta[s] ~ normal(0.0, 1.0);
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
