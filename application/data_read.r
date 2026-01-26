############################################################
# Weighted bootstrap + correct scaling order
############################################################
rm(list = ls())
.libPaths("./sims_R/rlib")
library(tidyverse)
set.seed(0)

datobj <- readRDS("MGL1704-hourly-paper-new.RDS")
datobj$X <- datobj$X[, !colnames(datobj$X) %in% c("b1", "b2"), drop = FALSE]

nsize <- 500
T <- length(datobj$ylist)
D <- ncol(datobj$ylist[[1]])
p <- ncol(datobj$X)

y_arr      <- array(NA, dim = c(T, nsize, D))
x_arr      <- array(NA, dim = c(T, nsize, p))
weight_arr <- matrix(NA, nrow = T, ncol = nsize)

############################################################
# Weighted bootstrap sampling
############################################################
for (tt in 1:T) {

    curr_y <- datobj$ylist[[tt]]
    curr_w <- datobj$countslist[[tt]]
    curr_size <- nrow(curr_y)

    p_prob <- curr_w / sum(curr_w)

    selected_idx <- sample(
        1:curr_size,
        size    = nsize,
        replace = TRUE,
        prob    = p_prob
    )

    # Y
    y_arr[tt,,] <- curr_y[selected_idx, ]

    # X (shared within time tt)
    x_arr[tt,,] <- matrix(
        rep(datobj$X[tt,], nsize),
        nrow = nsize,
        byrow = TRUE
    )

    # weights
    weight_arr[tt,] <- curr_w[selected_idx]
}

############################################################
# *** Correct scaling order ***
# must match python reshape: X.reshape(T*N, p)
############################################################

# DO NOT use byrow=TRUE -- that breaks order!
X_mat <- matrix(x_arr, nrow = T * nsize, ncol = p, byrow = FALSE)

# scaling
X_scaled <- scale(X_mat)

# reshape back to (T, nsize, p)
x_arr_scaled <- array(X_scaled, dim = c(T, nsize, p))

############################################################
# Save
############################################################
saveRDS(y_arr,           "y_arr.rds")
saveRDS(x_arr_scaled,    "x_arr.rds")
saveRDS(weight_arr,      "weight_arr.rds")
saveRDS(rownames(datobj$X), "datetime.rds")

############################################################
# Check order correctness
############################################################
# Check a few entries manually
cat("Before scaling X_arr[1,1,]:\n")
print(datobj$X[1,])

cat("After scaling (first row):\n")
print(x_arr_scaled[1,1,])

