# basic stan tutorial, following 
# https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started#how-to-use-rstan

library(rstan)
library(reshape2)
rstan_options(auto_write=TRUE)
options(mc.cores=parallel::detectCores())

set.seed(39)
#super basic example: linear regression with homoskedastic errors
#Y~N(mu+Xb,sigma2)
# X is an 50x9 matrix where each column is gaussian(5,sd=3)
X<-replicate(9,rnorm(50,5,3))
colMeans(X)
apply(X,2,sd)
#true parameter values:

mu<-33
b<-c(-3,-1.5,-.5,-.25,-0.01,0.01,.5,1,1.5) 
#strong negative: 1,2
#weak negative: 3,4
#insignificant effects: 5,6
#weak positive: 7
#strong positive: 8,9
sigma2<-4 #fairly high level of noise
y<-rnorm(50,mu+X%*%b,sqrt(sigma2))
frq<-lm(y~.,data=dat)
summary(frq) #standard approach accurately recovers true parameter values.
frq_params<-c(coef(frq),summary(frq)$sigma)
N<-nrow(X)
K<-ncol(X)
true_params<-c(mu,b,sqrt(sigma2))
frq_err<-frq_params-true_params
frq_rel_err<-frq_err/true_params

#run the stan program for linear regression. 
#no need for data in list, stan searches calling environment
lrfit<-stan(file="linreg.stan",iter=1000,chains=4) #takes about 7 seconds on my machine
print(lrfit) #lrfit is S4 object of class stanfit
help("stanfit-class")
#get posterior means
summary(lrfit)
stan_params<-summary(lrfit)$summary
stan_signif<-stan_params[,"2.5%"]*stan_params[,"97.5%"]>0
#stan found 4,5,6 to be not significant from zero versus true value being 5,6.
#the frequentist approach (OLS) made the same mistake (see above).
#compare stan values and frequentist values to true values

stan_params<-stan_params[-12,1]
stan_err<-stan_params-true_params
stan_rel_err<-stan_err/true_params

rel_err<-data.frame(var=names(stan_params),OLS=frq_rel_err,STAN=stan_rel_err)
plot_dat<-melt(rel_err,variable.name="method",id.vars="var",value.name="rel_err")
ggplot(data=plot_dat,aes(var,rel_err))+geom_bar(aes(fill=method),position="dodge",stat="identity")+theme_bw()
ggplot(data=plot_dat[-c(6,7,17,18),],aes(var,rel_err))+geom_bar(aes(fill=method),position="dodge",stat="identity")+theme_bw()

mse_frq<-sum(frq_err^2)/11
mse_stan<-sum(stan_err^2)/11
mse_frq
mse_stan

#we see that the stan method gives very similar results to the OLS method due to the uninformative prior.