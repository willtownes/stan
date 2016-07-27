data {
  int<lower=1> nvals;
  int<lower=1> L;
  int<lower=1,upper=nvals> N;
  int<lower=1,upper=nvals> G;
  int<lower=1,upper=N> nn[nvals];
  int<lower=1,upper=G> gg[nvals];
  real y[nvals];
}
parameters {
  vector[L] u[N];
  vector[L] v[G];
}
model {
  // prior
  for (n in 1:N) {
    u[n] ~ normal(0,1);
  }
  for (g in 1:G) {
    v[g] ~ normal(0,1);
  }
  // likelihood
  for (val in 1:nvals) {
    y[val] ~ normal(u[nn[val]]'*v[gg[val]],1);
  }
}