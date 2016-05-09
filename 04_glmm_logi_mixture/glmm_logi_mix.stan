data {
  real<lower=0> a; //concentration parameter of dirichlet prior
  int<lower=0> N; #number of total observations
  int<lower=1> D; #dimensionality of covariates, excluding intercept
  int<lower=1> K; #number of clusters
  int<lower=1> J; #number of mixture components
  matrix[N,D] X; #covariate design matrix
  int<lower=0,upper=1> y[N]; #binary outcome
  int<lower=1,upper=K> id[N]; #cluster id
}
transformed data {
  vector<lower=0>[J] alphavec; //vector of Dirichlet params
  alphavec <- rep_vector(a / J,J); //strange that rep_array failed here
}
parameters {
  real fixed_int; #intercept 
  vector[D] beta; #fixed effects
  vector[K] rand_ints;
  simplex[J] theta; //mixing proportions
  real mu[J]; // cluster means
  real<lower=0> sigma[J]; //scale parameters of mixture components
}
model {
  real soft_z[J]; //move to params block if assigning soft cluster IDs
  vector[N] eta;
  // priors
  theta ~ dirichlet(alphavec); //mixing probabilities
  mu ~ cauchy(0,5);
  sigma ~ cauchy(0,1);
  #for(d in 1:D) beta[d]~cauchy(0,5);
  beta~cauchy(0,5);
  #mixture prior for random effects
  for(k in 1:K){
    for(j in 1:J) {
      soft_z[j] <- dirichlet_log(theta,alphavec) + normal_log(rand_ints[k],mu[j],sigma[j]);
    }
    increment_log_prob(log_sum_exp(soft_z));
  }
  for(n in 1:N){
    eta[n]<-fixed_int + X[n]*beta + rand_ints[id[n]];
  }
  y ~ bernoulli_logit(eta);
}