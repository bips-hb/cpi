library(simsurv)

##' Illustration of survival CPI on simulated survival data

#' Idea: Illustrate an exemplary case of CPI computed on survival data
#' Exemplary case: treatment variable is affecting the event probability heavily 
#' in the short run, but influence declines as time increases (e.g. event is recovery
#' of a disease and treatment is drug that only works in the early stages of the disease)
#' 
#' \case{simsurv} package is used to simulate survival data with one binary treatment 
#' variable (sampled from binomial distribution), observations are censored for t*>60,
#' relationship between hazard rate and covariate over time is modeled by Gaussian function


# set seed for sampling
set.seed(9898)

# create dataframe of covariates (one treatment variable & one id variable)
cov <- data.frame(id = 1:10000, trt = rbinom(10000, 1, 0.5))

# simulate survival data
dat <- simsurv(dist = "weibull", lambdas = 0.1, gammas = 1.5, 
               betas = c(trt = -0.5), x = cov, tde = c(trt = 0.15),
               tdefunction = function(t) 39.23111*exp(-(t-10.91841)^2/(2*5.779492^2)), 
               maxt = 60)
dat <- merge(cov, dat)
head(dat)

# covert simulated survival data to mlr3 survival task
sim <- TaskSurv$new(
  id = "id",
  dat,
  time = "eventtime",
  event = "status",
  time2,
  type = c("right"),
  label = NA_character_
)

# compute CPI 
matrix.cpi = cpi(task = sim, 
                 learner = lrn("surv.coxph"), 
                 resampling = rsmp("holdout"), 
                 measure = "surv.graf",
                 test = "t", 
                 times = seq(1,100,by=0.05),
                 adjust = TRUE, 
                 method = "BH")


# plot cpi of treatment variable
plot_surv_cpi(matrix.cpi, CI = TRUE, varList = c("trt")) 











