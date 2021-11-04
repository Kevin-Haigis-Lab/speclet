CERES Mimic.

$$
\begin{aligned}
lfc &\sim q_s (h_g + d_{g,c} + \beta_c C) + \eta_b + o_s \\
o_s &\sim N[\mu_o, \sigma_o](\text{gene}) \\
\end{aligned}
$$

where:

- s: sgRNA
- g: gene
- c: cell line
- b: batch
- C: copy number (input data)

This is a mimic of the CERES model with an additional parameter for pDNA batch.
There are two optional parameters, `copynumber_cov` and `sgrna_intercept_cov`, to
control the inclusion of \(\beta_c\) and \(o_s\), respectively.

---

SpecletOne Model.

$$
\begin{aligned}
lfc &\sim h_s + d_{s,c} + \beta_c C + \eta_b \\
h_s &\sim N[\mu_h, \sigma_h](\text{gene}) \\
d_{s,c} &\sim N[\mu_d, \sigma_d](\text{gene}|\text{cell line}) \\
\end{aligned}
$$

where:

- s: sgRNA
- g: gene
- c: cell line
- b: batch
- C: copy number (input data)

This model is based on the CERES model, but removes the multiplicative sgRNA
"activity" score due to issues of non-identifiability. Also, the consistent gene
effect parameter has been replaced with a coefficient for sgRNA with a pooling prior
per gene. Similarly, the cell-line specific gene effect coefficient \(d_{s,c}\)
has been extended with a similar structure.

---

SpecletTwo Model.

$$
lfc \sim \alpha_{g,c} + \eta_b
$$

where:

- g: gene
- c: cell line
- b: batch

This is a simple model with varying intercepts for [gene|cell line] and batch.

---

SpecletFour Model.

$$
lfc \sim h_g + d_{g,c} + \beta_c C + \eta_b
$$

where:

- g: gene
- c: cell line
- b: batch
- C: copy number (input data)

A simple model with a separate consistent gene effect \(h_g\) and cell-line
varying gene effect \(d_{g,c}\). The coefficient for copy number effect
\(\beta_c\) varies by cell line and is optional.

---

SpecletFive Model.

$$
lfc \sim i + a_g + d_c + h_{g,c} + j_b
$$

where:

- g: gene
- c: cell line
- b: batch

The model is relatively simple. It has a single global intercept \(i\) and two
varying intercepts for gene \(a_g\) and cell line \(d_c\) and for varying
effects per gene per cell line \(h_{g,c}\). Finally, there is a coefficient for
batch effect \(j_b\).

---

SpecletSix Model.

$$
\begin{aligned}
lfc &\sim i + a_s + d_c + h_{g,c} + j_b +
k_c C^{(c)} + n_g C^{(g)} + q_{g,l} R^{(g,l)} + m_{g,l} M \\
a_s &\sim N[μ_a, σ_a](\text{gene}) \\
d_c &\sim N[μ_d, σ_d](\text{lineage}) \text{ (if more than one lineage)} \\
j_b &\sim N[μ_j, σ_j](\text{source}) \text{ (if more than one source)}
\end{aligned}
$$

where:

- s: sgRNA
- g: gene
- c: cell line
- l: cell line lineage
- b: batch
- o: data source (Broad or Sanger)

Below is a description of each parameter in the model:

- \(a_s\): sgRNA effect with hierarchical level for gene (g)
- \(d_c\): cell line effect with hierarchical level for lineage (l; if more than
one is found)
- \(j_b\): data source (o; if more than one is found)
- \(k_c\): cell line effect of copy number ( \(C^{(c)}\): z-scaled per cell
line)
- \(n_g\): gene effect of copy number ( \(C^{(g)}\): z-scaled per gene)
- \(q_{g,l}\): RNA effect varying per gene and cell line lineage
( \(R^{(g,l)}\): z-scaled within each gene and lineage)
- \(m_{g,l}\): mutation effect varying per gene and cell line lineage
( \(M \in {0, 1}\))

---

## SpecletSeven Model.

A very deep hierarchical model that is, in part, meant to be a proof-of-concept for
constructing, fitting, and interpreting such a "tall" hierarchical model.

Below is a key for the subscripts used:

- $s$: sgRNA
- $g$: gene
- $c$: cell line
- $l$: cell line lineage

The likelihood is a normal distribution over the log-fold change values.

$$
\begin{aligned}
lfc &\sim N(\mu, \sigma) \\
\mu &= a_{s,c} j_b \\
\sigma &\sim HN(1) \\
a_{s,c} &\sim_{s,c} N(\mu_a, \sigma_a)_{s,c} \\
j_b &\sim_b N(\mu_j, \sigma_j) \\
\mu_j &\sim N(0, 0.2) \quad \sigma_j \sim HN(0.5) \\
\sigma_a &\sim_s HN(\sigma_{\sigma_a})_s \\
\sigma_{\sigma_a} &\sim HN(1) \\
\end{aligned}
$$

What looks like it would normally be found at the first level of the model is
actually at the second level as each parameter is on the "per-gene" level. For
simplicity the individual pieces are split up below.

$$
\mu_a = h_{g,c} + k_c C^{(c)} + n_g C^{(g)} + q_{g,l} R^{(g,l)} + m_{g,l} M \\
$$

The $h_{g,c}$ parameter measures the varying effect per gene and per cell line. The
cell line effect is modeled further as coming from a distribution for the lineage.
This lineage level is only included if there is more than one lineage.

$$
\begin{aligned}
h_{g,c} &\sim_{g,c} N(\mu_h, \sigma_h)_{g,c} \\
\mu_h &\sim_{g,l} N(\mu_{\mu_h}, \sigma_{\mu_h})_{g,l}
\quad \sigma_h \sim_c HN(\sigma_{\sigma_h})_c \\
\mu_{\mu_h} &\sim N(0,1) \quad \sigma_{\mu_h} \sim HN(1) \\
\sigma_{\sigma_h} &\sim HN(1) \\
\end{aligned}
$$

The $k_c$ parameter measures the cell line-specific copy number effect. The
covariate $C^{(c)}$ is the copy number alterations scaled per cell line.

$$
\begin{aligned}
k_c &\sim_c N(\mu_k, \sigma_k)_l \\
\mu_k &\sim_l N(\mu_{\mu_k}, \sigma_{\mu_k})
\quad \sigma_k \sim_l HN(\sigma_{\sigma_k}) \\
\mu_{\mu_k} &\sim N(0,1) \quad \sigma_{\mu_k} \sim(1)
\quad \sigma_{\sigma_k} \sim HN(1)
\end{aligned}
$$

The $n_g$ parameter measures the gene-specific copy number effect. The covariate
$C^{(g)}$ is the copy number alterations scaled per gene.

$$
\begin{aligned}
n_g &\sim_g N(\mu_n, \sigma_n) \\
\mu_n &\sim N(0,1) \quad \sigma_n \sim HN(1)
\end{aligned}
$$

The $q_{g,l}$ parameter measures the effect of RNA expression for each gene and
lineage combination. The RNA expression covariate $R^{(g,l)}$ is scaled per gene per
lineage.

$$
\begin{aligned}
q_{g,l} &\sim_{g,l} N(\mu_q, \sigma_q) \\
\mu_q &\sim N(0, 5) \quad \sigma \sim HN(5)
\end{aligned}
$$

The $m_{g,l}$ parameter measures the effect of a mutation to the gene represented by
the covariate $M$ that is a binary indicator variable where $M_{g,c}=1$ if gene $g$
in cell line $c$ is mutated.

$$
\begin{aligned}
m_{g,l} &\sim_{g,l} N(\mu_m, \sigma_m) \\
\mu_m &\sim_g N(\mu_{\mu_m}, \sigma_{\mu_m}) \quad \sigma_m \sim HN(5) \\
\mu_{\mu_m} &\sim N(0, 5) \quad \sigma_{\mu_m} \sim HN(5)
\end{aligned}
$$

---

## SpecletEight.

A negative binomial model of the read counts from the CRISPR screen data.

---
