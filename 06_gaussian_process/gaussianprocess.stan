data {
  # N1,xtrain,ytrain are observed data
  int<lower=1> N1;
  vector[N1] xtrain;
  vector[N1] ytrain;
  # N2,xtest are covariate values on which to do predictions
  int<lower=1> N2;
  vector[N2] xtest;
}
transformed data {
  int<lower=1> N;
  vector[N1+N2] x;
  N = N1+N2;
  for (n in 1:N1) x[n] = xtrain[n];
  for (n in 1:N2) x[N1+n] = xtest[n];
}
parameters {
  real beta; #linear regression slope
  real beta0; #intercept term
  #covariance function hyperparameters
  real<lower=0> eta_sq;
  real<lower=0> inv_rho_sq;
  real<lower=0> sigma_sq;
  #predictions
  vector[N2] ytest;
}
transformed parameters {
  real<lower=0> rho_sq;
  #vector[N1+N2] mu;
  rho_sq = inv(inv_rho_sq);
  #mu = x*beta;
}
model {
  vector[N] y;
  matrix[N1+N2, N1+N2] Sigma;
  for (n in 1:N1) y[n] = ytrain[n];
  for (n in 1:N2) y[N1+n] = ytest[n];
  
  // off-diagonal elements
  for(i in 1:(N-1)) {
    for(j in (i+1):N) {
      Sigma[i,j] = eta_sq*exp(-rho_sq*pow(x[i]-x[j],2));
      Sigma[j,i] = Sigma[i,j];
    }
  }
  // diagonal elements
  for (k in 1:N) Sigma[k,k] = eta_sq + sigma_sq;
  
  eta_sq ~ cauchy(0,5);
  inv_rho_sq ~ cauchy(0,5);
  sigma_sq ~ cauchy(0,5);
  beta ~ cauchy(0,5); #weak regularization of regression slope
  #implicit uniform prior on regression coefficients y0,beta
  y ~ multi_normal(beta0+x*beta, Sigma);
}
