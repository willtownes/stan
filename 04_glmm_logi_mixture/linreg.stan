data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    vector[N] y;
}
parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + X*beta, sigma);
}