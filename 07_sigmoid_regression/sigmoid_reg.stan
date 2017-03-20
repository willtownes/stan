// fits the model p(y=1)=lo+(hi-lo)*expit(a+b*x)
// logistic regression parameters intercept (a) and slope (b)
// lo, hi control false positive and false negative fractions
// setting lo=0, hi=1 results in standard logistic regression
data {
  int<lower=1> N;
  int<lower=0,upper=1> y[N];
  vector[N] x;
  real prior_mean;
  real<lower=0> prior_sd;
}
transformed data {
  #real<lower=0,upper=1> window;
  #window = hi-lo;
}
parameters {
  #real<lower=0,upper=1> kappa;
  real a;
  real b;
  real<lower=0,upper=0.5> lo;
  real<lower=0.5,upper=1> hi;
}
transformed parameters {
  real<lower=0,upper=1> window;
  window = hi - lo;
}
model {
  b ~ normal(prior_mean,prior_sd);
  for(n in 1:N){
    y[n] ~ bernoulli(lo+window*inv_logit(a+b*x[n]));
  }
}
