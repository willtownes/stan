data {
  // N1,xtrain,ytrain are observed data
  int<lower=1> N1;
  real xtrain[N1];
  vector[N1] ytrain;
  // N2,xtest are covariate values on which to do predictions
  int<lower=1> N2;
  real xtest[N2];
}
transformed data {
  real delta = 1e-9; // small diagonal term to make cholesky stable
  int<lower=1> N = N1+N2;
  real x[N];
  for (n in 1:N1) x[n] = xtrain[n];
  for (n in 1:N2) x[N1+n] = xtest[n];
}
parameters {
  real beta; //linear regression slope
  real beta0; //intercept term
  // cholesky transformed isotropic latent vars
  vector[N] eta;
  //covariance function hyperparameters
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x, alpha, rho);
    // diagonal elements
    for (n in 1:N) K[n, n] = K[n, n] + delta;
    L_K = cholesky_decompose(K);
    f = L_K*eta; 
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ std_normal();
  eta ~ std_normal();
  for(n in 1:N1) ytrain[n]~normal(beta0+xtrain[n]*beta+f[n], sigma);
}
generated quantities {
  vector[N2] ytest;
  for(n in 1:N2){
    ytest[n] = normal_rng(beta0+xtest[n]*beta+f[N1+n], sigma);
  }
}
