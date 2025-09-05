.libPaths("./sims_R/rlib")
library(ecp)
library(changepoint.np)


evaluate_change_points <- function(est_CP, true_CP, T) {
  tau <- T - 1
  num_CP <- length(est_CP)
  
  if (num_CP == 0) {
    dist_est_gt <- Inf
    dist_gt_est <- -Inf
    covering_metric <- 0
  } else {
    holder_est_gt <- sapply(true_CP, function(i) min(abs(est_CP - i)))
    dist_est_gt <- max(holder_est_gt)
    
    holder_gt_est <- sapply(est_CP, function(i) min(abs(true_CP - i)))
    dist_gt_est <- max(holder_gt_est)
    
    gt_CP_all <- c(1, true_CP, T + 1)
    est_CP_all <- c(1, est_CP, T + 1)
    
    gt_list  <- lapply(2:length(gt_CP_all),  function(i) seq(gt_CP_all[i-1], gt_CP_all[i]-1))
    est_list <- lapply(2:length(est_CP_all), function(i) seq(est_CP_all[i-1], est_CP_all[i]-1))
    
    covering_metric <- 0
    for (A in gt_list) {
      jaccard <- sapply(est_list, function(Ap) {
        inter <- length(intersect(A, Ap))
        union <- length(union(A, Ap))
        if (union == 0) return(0)
        inter / union
      })
      covering_metric <- covering_metric + length(A) * max(jaccard)
    }
    covering_metric <- covering_metric / (tau + 1)
  }
  
  abs_error <- abs(num_CP - length(true_CP))
  
  return(data.frame(
    abs_error = abs_error,
    dist_est_gt = dist_est_gt,
    dist_gt_est = dist_gt_est,
    covering_metric = covering_metric,
    est_CP = I(list(est_CP))
  ))
}


true_CP <- c(51, 101, 151)
T <- 200

results_list <- list()

for (rep in 0:0) {
  dat_y <- read.csv(paste0("./sims_py/reps_sim_dat/y_rep", rep, ".csv"))
  dat_x <- read.csv(paste0("./sims_py/reps_sim_dat/x_rep", rep, ".csv"))

  Y <- as.matrix(dat_y)
  X <- as.matrix(dat_x)

  Y_array <- array(t(Y), dim = c(3, 100, T))
  Y_array <- aperm(Y_array, c(3, 2, 1))

  X_array <- array(t(X), dim = c(3,100,T))
  X_array <- aperm(X_array, c(3,2,1))
  
  Y_mean <- t(sapply(1:T, function(t) colMeans(Y_array[t,,])))
  X_mean <- t(sapply(1:T, function(t) colMeans(X_array[t,,])))
  # Method 1: ECP  (#################No replicates.#################)
  # (Matteson, D. S., & James, N. A. (2014). A nonparametric approach for multiple change point analysis of multivariate data. Journal of the American Statistical Association, 109(505), 334–345.)
  ecp_res <- e.divisive(Y_mean, R=199, sig.lvl=0.05)
  ecp_est_CP <- ecp_res$estimates[-c(1, length(ecp_res$estimates))] 
  ecp_metrics <- evaluate_change_points(ecp_est_CP, true_CP, T)
  ecp_metrics$rep <- rep  
  ecp_metrics$method <- "ecp"
  
  # Method 2: Nonparametric Changepoint (#########Only for Univariate CPD##########)
  # Haynes, K., Fearnhead, P., & Eckley, I. A. (2017). A computationally efficient nonparametric approach for changepoint detection. Statistics and Computing, 27(5), 1293–1305.
  # library(changepoint.np)
  # cpt.np_res <- cpt.np(t(Y_mean), method="PELT")
  # opt <- selectModel(cpt.np_res)

  # Method 3: Regression breakpoints (#############Only for Univariate CPD#############)
  # Zeileis, A., Leisch, F., Hornik, K., & Kleiber, C. (2002). strucchange: An R package for testing for structural change in linear regression models. Journal of Statistical Software, 7(2), 1–38.
  # library(strucchange)

  # Method 4: Bayesian CPD
  # Barry, D., & Hartigan, J. A. (1993). A Bayesian analysis for change point problems. JASA, 88(421), 309–319.
  # Erdman, C., & Emerson, J. W. (2007). bcp: An R package for performing a Bayesian analysis of change point problems. Journal of Statistical Software, 23(3), 1–13.
  T <- dim(Y_array)[1]
  N <- dim(Y_array)[2]
  dy <- dim(Y_array)[3]
  dx <- dim(X_array)[3]

  Y_mean <- t(sapply(1:T, function(t) colMeans(Y_array[t,,])))
  X_mean <- t(sapply(1:T, function(t) colMeans(X_array[t,,])))

  residuals_mat <- matrix(NA, nrow=T, ncol=dy)
  for (j in 1:dy) {
    dat <- data.frame(y = Y_mean[,j], X_mean)
    colnames(dat)[2:(dx+1)] <- paste0("X", 1:dx)
    formula <- as.formula(paste("y ~", paste(colnames(dat)[2:(dx+1)], collapse=" + ")))
    fit <- lm(formula, data=dat)
    residuals_mat[,j] <- resid(fit)
  }

  resid_norm <- rowSums(residuals_mat^2)


  fit_bcp <- bcp(resid_norm, burnin=500, mcmc=5000)

  plot(fit_bcp, main="Bayesian CPD on residual norm (mean across individuals)")

  posterior_probs <- fit_bcp$posterior.prob
  bcp_est_CP <- order(posterior_probs, decreasing = T)[1:3]+1
  bcp_metrics <- evaluate_change_points(bcp_est_CP, true_CP, T)
  bcp_metrics$rep <- rep  
  bcp_metrics$method <- "bcp"



  results_list[[rep+1]] <- ecp_metrics
}

df_results <- do.call(rbind, results_list)
print(df_results)
