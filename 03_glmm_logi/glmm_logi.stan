data {
    int<lower=0> N; #number of total observations
    int<lower=1> D; #dimensionality of covariates, excluding intercept
    int<lower=1> K; #number of clusters
    matrix[N,D] X; #covariate design matrix
    int<lower=0,upper=1> y[N]; #binary outcome
    int<lower=1,upper=K> id[N]; #cluster id
}
parameters {
    real alpha; #intercept
    vector[D] beta; #fixed effects
    real<lower=0> sigma; #stdev of random intercepts
    vector[K] rand_ints;
}
model {
  vector[N] eta;
  for(d in 1:D) beta[d]~cauchy(0,5);
  # prior for random effect stdev
  sigma~cauchy(0,5);
  #prior for random effects
  for(k in 1:K){
    rand_ints[k]~normal(0,sigma);
  }
  for(n in 1:N){
    eta[n]<-alpha + X[n]*beta + rand_ints[id[n]];
  }
  y ~ bernoulli_logit(eta);
}