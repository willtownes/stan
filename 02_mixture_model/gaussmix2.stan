# a slightly more sophisticated model- we place a symmetric dirichlet prior on cluster probabilities. We are assuming dimensionality=1 though.
data {
  real<lower=0> a; //concentration parameter of dirichlet prior
  //setting a to a too-small number can cause things to break
  //larger a-value means more regularization of cluster mixing parameters.
  //set a=K for a uniform prior.
  int<lower=0> N;  // number of data points
  int<lower=1> K;  // number of clusters
  real y[N];  // observations
}
transformed data {
  real alpha;
  real ymean;
  real<lower=0> ysd;
  vector<lower=0>[K] alphavec; //vector of Dirichlet probabilities
  alpha = a; //weird: Stan doesn't allow assign alpha as data value directly
  ymean=mean(y); //empirical bayes hyperparameters
  ysd=sd(y);
  alphavec = rep_vector(alpha / K,K); //strange that rep_array failed here
}
parameters {
  simplex[K] theta; //mixing proportions
  real mu[K]; // cluster means
  real<lower=0> sigma[K]; //scale parameters of mixture components
}

// uncomment if we want to return soft cluster assignments
# transformed parameters {
#   real<upper=0> soft_z[N,K]; // log probabilities of cluster assignments
#   for (n in 1:N)
#     for (k in 1:K)
#       soft_z[n,k] = dirichlet_log(theta,alphavec) + normal_log(y[n],mu[k],sigma[k]);

# }

model {
  real soft_z[K]; //comment out if assigning soft cluster IDs
  // prior
  theta ~ dirichlet(alphavec); //mixing probabilities
  mu ~ normal(2.5,ysd); //automatic vectorization??
  sigma ~ cauchy(0,2.5); //how does stan know to constrain this >0?

  // likelihood- method if we instantiate soft cluster IDs
  # for (n in 1:N)
  #  increment_log_prob(log_sum_exp(soft_z[n,:]));

  //likelihood- method if we do not instantiate soft cluster IDs
  for(n in 1:N) {
    for(k in 1:K) {
      soft_z[k] = dirichlet_log(theta,alphavec) + normal_log(y[n],mu[k],sigma[k]);
    }
    //increment_log_prob(log_sum_exp(soft_z));
    target += log_sum_exp(soft_z);
  }
}