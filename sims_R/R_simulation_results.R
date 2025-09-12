.libPaths("./sims_R/rlib")
library(ecp)
library(bcp)
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
rep <- 0
results_list <- list()
reps <- 20
for (rep in 0:(reps-1)) {
  print(paste("Now doing",rep,"replicates"))
  dat_y <- read.csv(paste0("./sims_py/reps_sim_dat/y_rep", rep, ".csv"))
  dat_x <- read.csv(paste0("./sims_py/reps_sim_dat/x_rep", rep, ".csv"))

  Y <- as.matrix(dat_y)
  X <- as.matrix(dat_x)

  Y_array <- array(t(Y), dim = c(3, 100, T))
  Y_array <- aperm(Y_array, c(3, 2, 1))

  X_array <- array(t(X), dim = c(3,100,T))
  X_array <- aperm(X_array, c(3,2,1))

  T <- dim(Y_array)[1]
  N <- dim(Y_array)[2]
  dy <- dim(Y_array)[3]
  dx <- dim(X_array)[3]

  Y_stack <- matrix(NA, nrow=T*N, ncol=dy)
  X_stack <- matrix(NA, nrow=T*N, ncol=dx)
  time_index <- rep(1:T, each=N)

  row_id <- 1
  for (t in 1:T) {
    Y_stack[row_id:(row_id+N-1), ] <- Y_array[t,,]
    X_stack[row_id:(row_id+N-1), ] <- X_array[t,,]
    row_id <- row_id + N
  }

  residuals_mat <- matrix(NA, nrow=T*N, ncol=dy)
  for (j in 1:dy) {
    fit <- lm(Y_stack[,j] ~ X_stack)
    residuals_mat[,j] <- resid(fit)
  }
  resid_norm <- sapply(1:T, function(t) {
    idx <- which(time_index == t)
    sum(residuals_mat[idx, ]^2) 
  })
  Y_array[1, 1:3, ]
  X_array[1, 1:3, ]
  # print(resid_norm)
  # Method 1: ECP  (#################No replicates.#################)
  # (Matteson, D. S., & James, N. A. (2014). A nonparametric approach for multiple change point analysis of multivariate data. Journal of the American Statistical Association, 109(505), 334–345.)
  res_ecp <- e.divisive(matrix(resid_norm, ncol=1), R=499)
  est_cp_ecp <- res_ecp$estimates[-c(1,length(res_ecp$estimates))]
  eva_ecp <- evaluate_change_points(est_cp_ecp, true_CP, T)
  eva_ecp["rep"] <- rep
  eva_ecp["method"] <- "ecp"
  eva_ecp$est_CP <- as.character(eva_ecp$est_CP)
  
  # Method 2: Nonparametric Changepoint (#########Only for Univariate CPD##########)
  # Haynes, K., Fearnhead, P., & Eckley, I. A. (2017). A computationally efficient nonparametric approach for changepoint detection. Statistics and Computing, 27(5), 1293–1305.
  library(changepoint.np)

  cpt.np_res <- cpt.np(resid_norm, method="PELT")
  est_cp_cpt <- cpt.np_res@cpts[-length(cpt.np_res@cpts)]+1
  eva_cpt <- evaluate_change_points(est_cp_cpt, true_CP, T)
  eva_cpt["rep"] <- rep
  eva_cpt["method"] <- "nonpar"
  eva_cpt$est_CP <- as.character(eva_cpt$est_CP)
  # Method 3: Regression breakpoints (#############Only for Univariate CPD#############)
  # Zeileis, A., Leisch, F., Hornik, K., & Kleiber, C. (2002). strucchange: An R package for testing for structural change in linear regression models. Journal of Statistical Software, 7(2), 1–38.
  library(strucchange)
  bp <- breakpoints(resid_norm ~ 1)
  est_cp_bp <- bp$breakpoints+1
  eva_bp <- evaluate_change_points(est_cp_bp, true_CP, T)
  eva_bp["rep"] <- rep
  eva_bp["method"] <- "bp"
  eva_bp$est_CP <- as.character(eva_bp$est_CP)
  # Method 4: Bayesian CPD
  # Barry, D., & Hartigan, J. A. (1993). A Bayesian analysis for change point problems. JASA, 88(421), 309–319.
  # Erdman, C., & Emerson, J. W. (2007). bcp: An R package for performing a Bayesian analysis of change point problems. Journal of Statistical Software, 23(3), 1–13.
  fit_bcp <- bcp(resid_norm, burnin=500, mcmc=5000)
  posterior_probs <- fit_bcp$posterior.prob
  bcp_est_CP <- which(posterior_probs > 0.8) + 1
  eva_bcp <- evaluate_change_points(bcp_est_CP, true_CP, T)
  eva_bcp["rep"] <- rep
  eva_bcp["method"] <- "bcp"
  eva_bcp$est_CP <- as.character(eva_bcp$est_CP)
  # Summary
  results_list[[rep+1]] <- rbind(eva_ecp, eva_cpt, eva_bp, eva_bcp)
}

df_results <- do.call(rbind, results_list)
# print(df_results)
# write.csv(df_results, "./R_sims_summary.csv", row.names = F)
py_results <- read.csv("./sims_py/exp_results_combined.csv")
full_results <- rbind(df_results,py_results)
library(dplyr)

final_results <- full_results %>%
  select(-rep, -est_CP) %>%    
  group_by(method) %>%        
  summarise(across(everything(), mean, na.rm = TRUE))

print(final_results)

