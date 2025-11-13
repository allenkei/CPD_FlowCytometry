.libPaths(c("./sims_R/rlib", .libPaths()))
library(ecp)
library(changepoint.np)
library(strucchange)
library(bcp)

true_CP <- c(101, 201)
reps <- 50
results_list <- vector("list", reps)

evaluate_change_points <- function(est_cp, true_cp, T, tol = 10) {

  if (length(est_cp) == 0) {
    return(data.frame(
      est_CP = NA,
      cover_rate = 0,
      avg_dist = NA,
      FP = 0,
      FN = length(true_cp),
      abs_error = length(true_cp),
      dist_est_gt = Inf,
      dist_gt_est = Inf,
      covering_metric = 0,
      num_detected = 0
    ))
  }

  dist_mat <- abs(outer(est_cp, true_cp, "-"))
  min_dist_true <- apply(dist_mat, 2, min)
  min_dist_est <- apply(dist_mat, 1, min)

  cover_rate <- mean(min_dist_true <= tol)
  avg_dist <- mean(min_dist_true)

  FP <- sum(min_dist_est > tol)
  FN <- sum(min_dist_true > tol)
  abs_error <- abs(length(est_cp) - length(true_cp))

  dist_est_gt <- max(min_dist_true)
  dist_gt_est <- max(min_dist_est)

  compute_covering_metric <- function(est_cp, true_cp, T) {
    if (length(est_cp) == 0 || length(true_cp) == 0) return(0)
    gt_cp_all  <- c(1, true_cp, T + 1)
    est_cp_all <- c(1, est_cp, T + 1)
    gt_ranges  <- lapply(seq_len(length(gt_cp_all) - 1),
                         function(i) gt_cp_all[i]:(gt_cp_all[i + 1] - 1))
    est_ranges <- lapply(seq_len(length(est_cp_all) - 1),
                         function(i) est_cp_all[i]:(est_cp_all[i + 1] - 1))

    total <- 0
    for (A in gt_ranges) {
      jaccard_vals <- sapply(est_ranges, function(B) {
        inter <- length(intersect(A, B))
        union <- length(union(A, B))
        ifelse(union == 0, 0, inter / union)
      })
      total <- total + length(A) * max(jaccard_vals)
    }
    total / (T - 1)
  }

  covering_metric <- compute_covering_metric(est_cp, true_cp, T)

  data.frame(
    est_CP = paste(est_cp, collapse = ","),
    cover_rate = cover_rate,
    avg_dist = avg_dist,
    FP = FP,
    FN = FN,
    abs_error = abs_error,
    dist_est_gt = dist_est_gt,
    dist_gt_est = dist_gt_est,
    covering_metric = covering_metric,
    num_detected = length(est_cp)
  )
}



for (rep in 1:reps) {
  cat("Now doing", rep, "replicates\n")

  dat_y <- readRDS(paste0("./sims_py/real_data_sim/sim_dat_ult_5_", rep, ".RDS"))[,,1:3]  # (T,N,dy)
  dat_x <- readRDS(paste0("./sims_py/real_data_sim/sim_x_ult_5_", rep, ".RDS"))           # (T,dx)

  T <- dim(dat_y)[1]
  N <- dim(dat_y)[2]
  dy <- dim(dat_y)[3]
  dx <- ncol(dat_x)


  Y_stack <- matrix(aperm(dat_y, c(2,1,3)), nrow = T*N, ncol = dy)
  X_stack <- dat_x[rep(1:T, each = N), ]
  time_index <- rep(1:T, each = N)

  residuals_mat <- matrix(NA, nrow = T*N, ncol = dy)
  for (j in 1:dy) {
    fit <- lm(Y_stack[, j] ~ X_stack)
    residuals_mat[, j] <- resid(fit)
  }

  resid_norm <- sapply(1:T, function(t) {
    idx <- which(time_index == t)
    sum(residuals_mat[idx, ]^2)
  })



  ## Method 1: ecp
  resid_t <- t(sapply(1:T, function(t) {
  idx <- which(time_index == t)
  colMeans(residuals_mat[idx, , drop = FALSE]) 
  }))
  dim(resid_t)

  res_ecp <- e.divisive(resid_t, R = 999, sig.lvl = 0.01)
  est_cp_ecp <- res_ecp$estimates[-c(1, length(res_ecp$estimates))]
  eva_ecp <- evaluate_change_points(est_cp_ecp, true_CP, T)
  eva_ecp["rep"] <- rep; eva_ecp["method"] <- "ecp"
  eva_ecp$est_CP <- as.character(eva_ecp$est_CP)
  eva_ecp
  ## Method 2: changepoint.np
  cpt_res <- cpt.np(
    resid_norm,
    method = "PELT",
    penalty = "Manual",
    pen.value = 10 * log(N), 
    minseglen = 10
  )
  est_cp_cpt <- cpt_res@cpts[-length(cpt_res@cpts)] + 1
  eva_cpt <- evaluate_change_points(est_cp_cpt, true_CP, T)
  eva_cpt["rep"] <- rep; eva_cpt["method"] <- "nonpar"
  eva_cpt$est_CP <- as.character(eva_cpt$est_CP)
  eva_cpt

  ## Method 3: strucchange
  bp <- breakpoints(resid_norm ~ 1)
  est_cp_bp <- bp$breakpoints + 1
  eva_bp <- evaluate_change_points(est_cp_bp, true_CP, T)
  eva_bp["rep"] <- rep; eva_bp["method"] <- "bp"
  eva_bp$est_CP <- as.character(eva_bp$est_CP)


  ## Method 4: Bayesian
  fit_bcp <- bcp(resid_norm, burnin = 1000, mcmc = 10000, p0 = 0.00001)
  bcp_est_CP <- which(fit_bcp$posterior.prob > 0.999) + 1
  eva_bcp <- evaluate_change_points(bcp_est_CP, true_CP, T)
  eva_bcp["rep"] <- rep; eva_bcp["method"] <- "bcp"
  eva_bcp$est_CP <- as.character(eva_bcp$est_CP)
  eva_bcp
  results_list[[rep]] <- rbind(eva_ecp, eva_cpt, eva_bp, eva_bcp)
}

df_results <- do.call(rbind, results_list)
df_clean <- df_results[, c(
  "method", "FP", "FN", "abs_error",
  "dist_est_gt", "dist_gt_est",
  "covering_metric"
)]

write.csv(df_clean, "./sims_R/R_sims_summary.csv", row.names = FALSE)

print(head(df_clean))
summary_df <- aggregate(
  cbind(FP, FN, abs_error, dist_est_gt, dist_gt_est, covering_metric) ~ method,
  data = df_clean,
  FUN = mean,
  na.rm = TRUE
)
print(summary_df)

