#!/usr/bin/env Rscript
# Phase B — Multi-target analysis on unified CSV.
# Same paper methods: per-target spatial-block-CV RF, Welch's t (Grass vs
# Crop), bootstrap CIs.

suppressMessages({
  library(tidyverse); library(boot); library(ranger); library(car)
})
set.seed(42)

df <- read.csv("/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/soil_unified.csv")
ae <- sprintf("A%02d", 0:63)
df$log_soc <- log1p(df$soc)
df$ph <- df$ph_x10 / 10
logit_pct <- function(x) log((x/100 + 1e-3) / (1 - x/100 + 1e-3))
df$sand_logit <- logit_pct(df$sand_pct)
df$clay_logit <- logit_pct(df$clay_pct)
df$land_cover <- as.character(df$land_cover)

# Spatial-block folds (same recipe as paper)
df$block <- paste0(round(df$lon), "_", round(df$lat))
ub <- unique(df$block)
df$fold <- sample(1:5, length(ub), replace = TRUE)[match(df$block, ub)]

targets <- c("log_soc","ph","sand_logit","clay_logit","bd")

# ---------- per-target spatial-CV RF ----------
cat("============================================================\n")
cat("PHASE B: Per-target spatial-block-CV Random Forest\n")
cat("============================================================\n")
rf_results <- map_dfr(targets, function(t) {
  form <- as.formula(paste(t, "~", paste(ae, collapse = "+")))
  cv <- map_dfr(1:5, function(k) {
    tr <- df %>% filter(fold != k); te <- df %>% filter(fold == k)
    m  <- ranger(form, data = tr, num.trees = 500, seed = 42)
    pr <- predict(m, data = te)$predictions
    tibble(fold = k, n = nrow(te),
           rmse = sqrt(mean((pr - te[[t]])^2)),
           r2 = cor(pr, te[[t]])^2)
  })
  ci_b <- boot::boot(cv$r2, function(x, i) mean(x[i]), R = 10000)
  ci <- boot::boot.ci(ci_b, type = "bca")$bca[4:5]
  tibble(target = t, mean_r2 = mean(cv$r2), lo = ci[1], hi = ci[2],
         mean_rmse = mean(cv$rmse))
})
print(rf_results)

# ---------- per-target Welch's t (Grass vs Crop) ----------
cat("\n============================================================\n")
cat("PHASE B: Welch's two-sample t per target (Grass vs Crop)\n")
cat("============================================================\n")
gg <- df %>% filter(land_cover %in% c("10","12"))
ht_results <- map_dfr(targets, function(t) {
  res <- t.test(reformulate("land_cover", t), data = gg)
  g1 <- gg[[t]][gg$land_cover == "10"]
  g2 <- gg[[t]][gg$land_cover == "12"]
  psd <- sqrt(((length(g1)-1)*var(g1) + (length(g2)-1)*var(g2)) /
              (length(g1)+length(g2)-2))
  d   <- (mean(g1) - mean(g2)) / psd
  lev <- car::leveneTest(reformulate("factor(land_cover)", t), data = gg)
  tibble(target = t,
         t_stat = res$statistic, df = res$parameter,
         p_value = res$p.value,
         ci_lo = res$conf.int[1], ci_hi = res$conf.int[2],
         cohens_d = d,
         levene_p = lev[["Pr(>F)"]][1])
})
print(ht_results)

# ---------- top correlations per target ----------
cat("\n============================================================\n")
cat("PHASE B: Top-3 |r| AlphaEarth dim per target\n")
cat("============================================================\n")
top_per_target <- map_dfr(targets, function(t) {
  rs <- sapply(ae, function(c) cor(df[[c]], df[[t]]))
  tibble(target = t,
         top1 = sprintf("%s (r=%+.3f)", names(rs)[order(-abs(rs))[1]], rs[order(-abs(rs))[1]]),
         top2 = sprintf("%s (r=%+.3f)", names(rs)[order(-abs(rs))[2]], rs[order(-abs(rs))[2]]),
         top3 = sprintf("%s (r=%+.3f)", names(rs)[order(-abs(rs))[3]], rs[order(-abs(rs))[3]]))
})
print(top_per_target, width = Inf)

# ---------- Save figures ----------
dir.create("/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/figures",
           showWarnings = FALSE, recursive = TRUE)
fig_dir <- "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/figures"

# 5-panel pred-vs-obs
oof_per_target <- map_dfr(targets, function(t) {
  form <- as.formula(paste(t, "~", paste(ae, collapse = "+")))
  rows <- map_dfr(1:5, function(k) {
    tr <- df %>% filter(fold != k); te <- df %>% filter(fold == k)
    m  <- ranger(form, data = tr, num.trees = 500, seed = 42)
    pr <- predict(m, data = te)$predictions
    tibble(target = t, pred = pr, obs = te[[t]])
  })
  rows
})
ord <- rf_results %>% arrange(desc(mean_r2)) %>% pull(target)
oof_per_target$target <- factor(oof_per_target$target, levels = ord)
target_label <- c(log_soc="log(1+SOC)", ph="pH", sand_logit="logit(sand%)",
                  clay_logit="logit(clay%)", bd="Bulk density")
oof_per_target$lab <- target_label[as.character(oof_per_target$target)]
oof_per_target$lab <- factor(oof_per_target$lab,
                             levels = target_label[ord])

p <- ggplot(oof_per_target, aes(obs, pred)) +
  geom_point(alpha = 0.25, size = 0.5, color = "#5b8def") +
  geom_abline(slope = 1, intercept = 0, color = "#c0392b", linetype = "dashed") +
  facet_wrap(~ lab, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(x = "Observed", y = "Predicted",
       title = "Out-of-fold predicted vs. observed (5-fold spatial-block CV)")
ggsave(file.path(fig_dir, "phaseB_pred_obs.png"),
       p, width = 9, height = 5, dpi = 300)

# 5x64 correlation barplot
corr_long <- map_dfr(targets, function(t) {
  rs <- sapply(ae, function(c) cor(df[[c]], df[[t]]))
  tibble(target = t, dim = ae, r = rs)
})
corr_long$lab <- target_label[corr_long$target]
p2 <- ggplot(corr_long, aes(reorder(dim, r), r, fill = r > 0)) +
  geom_col() +
  facet_wrap(~ lab, ncol = 1, scales = "free_y") +
  scale_fill_manual(values = c("#c0392b","#2980b9"), guide = "none") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 90, size = 4)) +
  labs(x = NULL, y = "Pearson r")
ggsave(file.path(fig_dir, "phaseB_corr_bar.png"),
       p2, width = 9, height = 11, dpi = 300)

# Save numeric results
write.csv(rf_results,
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseB_rf_results.csv",
          row.names = FALSE)
write.csv(ht_results,
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseB_ht_results.csv",
          row.names = FALSE)
write.csv(top_per_target,
          "/Users/lukerobinson/Dropbox/school/stat_5000/project/future_work/data/phaseB_top_dims.csv",
          row.names = FALSE)
cat("\nSaved figures + CSVs to ../figures/ and ../data/\n")
