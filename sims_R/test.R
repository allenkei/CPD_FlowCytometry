library(ggplot2)

# 假设已经读进来 y_rep0.csv 并 reshape 成 Y_array: (200,100,3)
dat_y <- read.csv("./reps_sim_dat/y_rep0.csv")
Y <- as.matrix(dat_y)

T <- 200
N <- 100
D <- 3
Y_array <- array(t(Y), dim = c(D, N, T))   # (3,100,200)
Y_array <- aperm(Y_array, c(3,2,1))        # (200,100,3)

# 随机选 8 个 z
set.seed(123)
sel_idx <- sample(1:N, 8)

# 手工转成长数据框
df_list <- list()
for (z in sel_idx) {
  for (d in 1:D) {
    df_tmp <- data.frame(
      time = 1:T,
      value = Y_array[, z, d],
      dimension = paste0("y", d),
      series = paste0("z", z)
    )
    df_list[[length(df_list) + 1]] <- df_tmp
  }
}
df_plot <- do.call(rbind, df_list)

# 检查结构
str(df_plot)
head(df_plot)

ggplot(df_plot, aes(x = time, y = value, color = dimension)) +
  geom_line() +
  facet_wrap(~series, ncol = 2, scales = "free_y") +
  labs(title = "Random 8 trajectories of Y (3 dims)",
       x = "Time", y = "Value", color = "Dimension") +
  theme_minimal()
