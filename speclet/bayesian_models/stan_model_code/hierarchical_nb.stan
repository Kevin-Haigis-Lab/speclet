// Hierarchical negative binomial model.

data {
    int<lower=0> N;  // number of data points
    int<lower=1> S;  // number of sgRNAs
    int<lower=1> G;  // number of genes
    int<lower=1> C;  // number of cell lines
    int<lower=1> L;  // number of lineages
    array[N] real<lower=0> ct_initial;
    array[N] int<lower=0> ct_final;
    array[N] int<lower=1, upper=S> sgrna_idx;
    array[N] int<lower=1, upper=G> gene_idx;
    array[N] int<lower=1, upper=C> cellline_idx;
    array[N] int<lower=1, upper=L> lineage_idx;

}

parameters {
    real z;
    real<lower=0> sigma_a;
    array[S] real a;
    real<lower=0> sigma_b;
    array[C] real delta_b;
    real<lower=0> sigma_d;
    array[G,L] real delta_d;
    real<lower=0> alpha_alpha;
    real<lower=0> beta_alpha;
    array[G] real<lower=0> alpha;
}

transformed parameters {
    array[N] real eta;
    array[N] real<lower=0> mu;
    array[C] real b;
    array[G,L] real d;

    for (g in 1:G) {
        for (l in 1:L) {
            d[g,l] = 0.0 + delta_d[g,l] * sigma_d;
        }
    }

    for (c in 1:C) {
        b[c] = 0.0 + delta_b[c] * sigma_b;
    }

    for (n in 1:N) {
        eta[n] = z + a[sgrna_idx[n]] + b[cellline_idx[n]] + d[gene_idx[n]][lineage_idx[n]];
        mu[n] = exp(eta[n]) * ct_initial[n];
    }
}

model {
    // Priors
    sigma_a ~ normal(0.0, 2.5);
    sigma_b ~ normal(0.0, 2.5);
    sigma_d ~ normal(0.0, 2.5);

    z ~ normal(0.0, 5.0);
    a ~ normal(0.0, sigma_a);
    delta_b ~ normal(0.0, 1.0);

    alpha_alpha ~ gamma(2.0, 0.5);
    beta_alpha ~ gamma(2.0, 0.5);

    for (g in 1:G) {
        delta_d[g] ~ normal(0.0, 1.0);
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
