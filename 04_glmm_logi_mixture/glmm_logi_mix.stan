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
  real alpha;
  real<lower=0> ysd;
  vector<lower=0>[K] alphavec; //vector of Dirichlet probabilities
  alpha <- a; //weird: Stan doesn't allow assign alpha as data value directly
  ymean<-mean(y); //empirical bayes hyperparameters
  ysd<-sd(y);
  alphavec <- rep_vector(alpha / K,K); //strange that rep_array failed here
}
parameters {
    real fixed_int; #intercept
    vector[D] beta; #fixed effects
    vector[K] rand_ints;
    
}
model {
  real soft_z[J]; //comment out if assigning soft cluster IDs
  vector[N] eta;
  
  // priors
  theta ~ dirichlet(alphavec); //mixing probabilities
  mu ~ cauchy(0,5);
  sigma ~ cauchy(0,2);
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