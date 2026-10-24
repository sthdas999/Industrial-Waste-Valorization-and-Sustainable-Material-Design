# ============================================================
# REAL-DATA CASE STUDY:
# UCI Concrete Compressive Strength Dataset
# Yeh (1998), UCI Machine Learning Repository, DOI: 10.24432/C5PK67
# ============================================================

# -----------------------------
# 1. Install and load packages
# -----------------------------

required_packages <- c(
  "readxl", "ggplot2", "dplyr", "tidyr",
  "corrplot", "randomForest", "e1071",
  "gbm", "nnet", "car"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace,
          logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, dependencies = TRUE)
}

library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(randomForest)
library(e1071)
library(gbm)
library(nnet)
library(car)

set.seed(20261024)

# -----------------------------
# 2. Download and read the data
# -----------------------------

data_dir <- "uci_concrete_case_study"
dir.create(data_dir, showWarnings = FALSE)

zip_file <- file.path(data_dir, "concrete_strength.zip")

data_url <- paste0(
  "https://archive.ics.uci.edu/static/public/",
  "165/concrete+compressive+strength.zip"
)

if (!file.exists(zip_file)) {
  download.file(
    url = data_url,
    destfile = zip_file,
    mode = "wb",
    method = "libcurl"
  )
}

unzip(zip_file, exdir = data_dir)

excel_file <- list.files(
  data_dir,
  pattern = "\\.xls$|\\.xlsx$",
  recursive = TRUE,
  full.names = TRUE,
  ignore.case = TRUE
)

if (length(excel_file) == 0) {
  stop("No Excel dataset was found in the downloaded archive.")
}

raw_data <- readxl::read_excel(excel_file[1])

# Rename columns by position to avoid inconsistencies in
# the original column headings.
if (ncol(raw_data) != 9) {
  stop("Expected 9 columns: 8 predictors and 1 response.")
}

names(raw_data) <- c(
  "cement",
  "slag",
  "flyash",
  "water",
  "superplasticizer",
  "coarse_aggregate",
  "fine_aggregate",
  "age",
  "strength"
)

dat <- as.data.frame(raw_data)

# Ensure all columns are numeric.
dat[] <- lapply(dat, function(x) as.numeric(as.character(x)))

# Basic checks.
cat("\nDataset dimensions:\n")
print(dim(dat))

cat("\nColumn names:\n")
print(names(dat))

cat("\nMissing values by variable:\n")
print(colSums(is.na(dat)))

cat("\nDuplicate rows:\n")
print(sum(duplicated(dat)))

# Remove incomplete records only if any are present.
dat <- dat[complete.cases(dat), ]

# Export the analysis-ready data.
write.csv(
  dat,
  file.path(data_dir, "analysis_ready_data.csv"),
  row.names = FALSE
)

# -----------------------------
# 3. Descriptive statistics
# -----------------------------

describe_variable <- function(x) {
  c(
    N = sum(!is.na(x)),
    Mean = mean(x, na.rm = TRUE),
    SD = sd(x, na.rm = TRUE),
    Minimum = min(x, na.rm = TRUE),
    Q1 = unname(quantile(x, 0.25, na.rm = TRUE)),
    Median = median(x, na.rm = TRUE),
    Q3 = unname(quantile(x, 0.75, na.rm = TRUE)),
    Maximum = max(x, na.rm = TRUE)
  )
}

descriptive_table <- as.data.frame(
  t(vapply(dat, describe_variable, numeric(8)))
)

descriptive_table$Variable <- rownames(descriptive_table)
rownames(descriptive_table) <- NULL

descriptive_table <- descriptive_table %>%
  select(Variable, everything())

cat("\nDescriptive statistics:\n")
print(descriptive_table, digits = 4)

write.csv(
  descriptive_table,
  file.path(data_dir, "descriptive_statistics.csv"),
  row.names = FALSE
)

# Strength-only summary.
strength_summary <- data.frame(
  N = length(dat$strength),
  Mean = mean(dat$strength),
  SD = sd(dat$strength),
  Minimum = min(dat$strength),
  Q1 = unname(quantile(dat$strength, 0.25)),
  Median = median(dat$strength),
  Q3 = unname(quantile(dat$strength, 0.75)),
  Maximum = max(dat$strength)
)

cat("\nCompressive-strength summary:\n")
print(strength_summary, digits = 4)

# -----------------------------
# 4. Histograms and boxplots
# -----------------------------

p_hist <- ggplot(dat, aes(x = strength)) +
  geom_histogram(
    bins = 30,
    colour = "black",
    fill = "grey75"
  ) +
  labs(
    title = "Distribution of concrete compressive strength",
    x = "Compressive strength (MPa)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(data_dir, "strength_histogram.png"),
  p_hist, width = 7, height = 5, dpi = 300
)

p_box <- ggplot(dat, aes(y = strength)) +
  geom_boxplot(fill = "grey80") +
  labs(
    title = "Boxplot of compressive strength",
    x = NULL,
    y = "Compressive strength (MPa)"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(data_dir, "strength_boxplot.png"),
  p_box, width = 5, height = 5, dpi = 300
)

# -----------------------------
# 5. Correlation analysis
# -----------------------------

cor_matrix <- cor(
  dat,
  use = "pairwise.complete.obs",
  method = "pearson"
)

cat("\nPearson correlation matrix:\n")
print(round(cor_matrix, 3))

write.csv(
  cor_matrix,
  file.path(data_dir, "pearson_correlation_matrix.csv")
)

png(
  file.path(data_dir, "correlation_heatmap.png"),
  width = 2400, height = 2000, res = 300
)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  number.cex = 0.55,
  tl.col = "black",
  tl.srt = 45,
  diag = FALSE
)

dev.off()

# Optional Spearman correlation matrix.
spearman_matrix <- cor(
  dat,
  use = "pairwise.complete.obs",
  method = "spearman"
)

write.csv(
  spearman_matrix,
  file.path(data_dir, "spearman_correlation_matrix.csv")
)

# -----------------------------
# 6. Fly ash presence comparison
# -----------------------------

dat$flyash_group <- factor(
  ifelse(dat$flyash == 0, "Zero fly ash", "Positive fly ash"),
  levels = c("Zero fly ash", "Positive fly ash")
)

flyash_summary <- dat %>%
  group_by(flyash_group) %>%
  summarise(
    N = n(),
    Mean = mean(strength),
    SD = sd(strength),
    Median = median(strength),
    .groups = "drop"
  )

cat("\nFly ash group summary:\n")
print(flyash_summary)

write.csv(
  flyash_summary,
  file.path(data_dir, "flyash_group_summary.csv"),
  row.names = FALSE
)

# Exploratory one-way ANOVA.
flyash_aov <- aov(strength ~ flyash_group, data = dat)

cat("\nANOVA: zero versus positive fly ash:\n")
print(summary(flyash_aov))

capture.output(
  summary(flyash_aov),
  file = file.path(data_dir, "flyash_anova.txt")
)

p_flyash <- ggplot(
  dat,
  aes(x = flyash_group, y = strength)
) +
  geom_boxplot(fill = "grey80") +
  labs(
    title = "Strength by fly ash presence",
    x = "Fly ash group",
    y = "Compressive strength (MPa)"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(data_dir, "flyash_group_boxplot.png"),
  p_flyash, width = 7, height = 5, dpi = 300
)

# -----------------------------
# 7. Training/test split
# -----------------------------

# A fixed random 80/20 split.
# This is observation-level validation, not batch-wise validation.
set.seed(20261024)

n <- nrow(dat)
train_id <- sample(seq_len(n), size = floor(0.80 * n))

train <- dat[train_id, ]
test <- dat[-train_id, ]

predictor_names <- c(
  "cement", "slag", "flyash", "water",
  "superplasticizer", "coarse_aggregate",
  "fine_aggregate", "age"
)

# -----------------------------
# 8. Fit statistical models
# -----------------------------

fit_linear <- lm(
  strength ~ cement + slag + flyash + water +
    superplasticizer + coarse_aggregate +
    fine_aggregate + age,
  data = train
)

fit_quadratic <- lm(
  strength ~ cement + slag + flyash + I(flyash^2) +
    water + superplasticizer + coarse_aggregate +
    fine_aggregate + age + I(age^2),
  data = train
)

fit_rsm <- lm(
  strength ~ cement + slag + flyash + water +
    superplasticizer + coarse_aggregate +
    fine_aggregate + age +
    I(flyash^2) + I(age^2) + flyash:age,
  data = train
)

cat("\nLinear regression summary:\n")
print(summary(fit_linear))

cat("\nQuadratic regression summary:\n")
print(summary(fit_quadratic))

cat("\nResponse-surface model summary:\n")
print(summary(fit_rsm))

# Regression diagnostics for the linear model.
png(
  file.path(data_dir, "linear_model_diagnostics.png"),
  width = 2400, height = 2000, res = 300
)
par(mfrow = c(2, 2))
plot(fit_linear)
dev.off()

# Optional multicollinearity assessment.
cat("\nVIF values for the linear model:\n")
print(car::vif(fit_linear))

# -----------------------------
# 9. Standardize predictors
#    using training data only
# -----------------------------

x_train <- as.matrix(train[, predictor_names])
x_test <- as.matrix(test[, predictor_names])

x_mean <- colMeans(x_train)
x_sd <- apply(x_train, 2, sd)

# Protect against any constant predictor.
x_sd[x_sd == 0] <- 1

x_train_scaled <- scale(
  x_train,
  center = x_mean,
  scale = x_sd
)

x_test_scaled <- scale(
  x_test,
  center = x_mean,
  scale = x_sd
)

y_train <- train$strength

# -----------------------------
# 10. Fit machine-learning models
# -----------------------------

# Random forest.
set.seed(20261024)

fit_rf <- randomForest(
  x = train[, predictor_names],
  y = train$strength,
  ntree = 800,
  importance = TRUE
)

pred_rf <- predict(
  fit_rf,
  newdata = test[, predictor_names]
)

# Support vector regression.
set.seed(20261024)

fit_svr <- e1071::svm(
  x = x_train_scaled,
  y = y_train,
  type = "eps-regression",
  kernel = "radial",
  scale = FALSE
)

pred_svr <- predict(
  fit_svr,
  newdata = x_test_scaled
)

# Artificial neural network.
set.seed(20261024)

fit_ann <- nnet::nnet(
  x = x_train_scaled,
  y = y_train,
  size = 8,
  linout = TRUE,
  decay = 0.01,
  maxit = 1000,
  trace = FALSE
)

pred_ann <- as.numeric(
  predict(fit_ann, newdata = x_test_scaled)
)

# Gradient boosting.
set.seed(20261024)

fit_gbm <- gbm::gbm(
  strength ~ cement + slag + flyash + water +
    superplasticizer + coarse_aggregate +
    fine_aggregate + age,
  data = train,
  distribution = "gaussian",
  n.trees = 1500,
  interaction.depth = 3,
  shrinkage = 0.03,
  n.minobsinnode = 10,
  bag.fraction = 0.8,
  train.fraction = 1.0,
  verbose = FALSE
)

pred_gbm <- predict(
  fit_gbm,
  newdata = test,
  n.trees = 1500
)

# -----------------------------
# 11. Common test-set metrics
# -----------------------------

calculate_metrics <- function(observed, predicted) {
  rmse <- sqrt(mean((observed - predicted)^2))
  mae <- mean(abs(observed - predicted))
  test_r2 <- 1 -
    sum((observed - predicted)^2) /
    sum((observed - mean(observed))^2)
  
  c(
    RMSE = rmse,
    MAE = mae,
    Test_R2 = test_r2
  )
}

predictions <- list(
  "Linear regression" = predict(fit_linear, newdata = test),
  "Quadratic regression" = predict(fit_quadratic, newdata = test),
  "Response-surface model" = predict(fit_rsm, newdata = test),
  "Random forest" = pred_rf,
  "SVR" = pred_svr,
  "ANN" = pred_ann,
  "Gradient boosting" = pred_gbm
)

model_comparison <- do.call(
  rbind,
  lapply(names(predictions), function(model_name) {
    metrics <- calculate_metrics(
      observed = test$strength,
      predicted = as.numeric(predictions[[model_name]])
    )
    
    data.frame(
      Model = model_name,
      RMSE = unname(metrics["RMSE"]),
      MAE = unname(metrics["MAE"]),
      Test_R2 = unname(metrics["Test_R2"])
    )
  })
)

model_comparison <- model_comparison[
  order(model_comparison$RMSE),
]

model_comparison$Rank <- seq_len(nrow(model_comparison))

cat("\nHeld-out test-set model comparison:\n")
print(model_comparison, row.names = FALSE, digits = 4)

write.csv(
  model_comparison,
  file.path(data_dir, "model_comparison.csv"),
  row.names = FALSE
)

# -----------------------------
# 12. Training-set model criteria
# -----------------------------

training_model_criteria <- data.frame(
  Model = c(
    "Linear regression",
    "Quadratic regression",
    "Response-surface model"
  ),
  Adjusted_R2 = c(
    summary(fit_linear)$adj.r.squared,
    summary(fit_quadratic)$adj.r.squared,
    summary(fit_rsm)$adj.r.squared
  ),
  AIC = c(
    AIC(fit_linear),
    AIC(fit_quadratic),
    AIC(fit_rsm)
  ),
  BIC = c(
    BIC(fit_linear),
    BIC(fit_quadratic),
    BIC(fit_rsm)
  )
)

cat("\nTraining-set model criteria:\n")
print(training_model_criteria, row.names = FALSE, digits = 4)

write.csv(
  training_model_criteria,
  file.path(data_dir, "training_model_criteria.csv"),
  row.names = FALSE
)

# -----------------------------
# 13. Observed versus predicted
# -----------------------------

prediction_data <- data.frame(
  Observed = test$strength,
  Linear = as.numeric(predictions[["Linear regression"]]),
  Quadratic = as.numeric(predictions[["Quadratic regression"]]),
  RSM = as.numeric(predictions[["Response-surface model"]]),
  RandomForest = as.numeric(predictions[["Random forest"]]),
  SVR = as.numeric(predictions[["SVR"]]),
  ANN = as.numeric(predictions[["ANN"]]),
  GradientBoosting = as.numeric(predictions[["Gradient boosting"]])
)

write.csv(
  prediction_data,
  file.path(data_dir, "test_set_predictions.csv"),
  row.names = FALSE
)

prediction_long <- prediction_data %>%
  pivot_longer(
    cols = -Observed,
    names_to = "Model",
    values_to = "Predicted"
  )

p_prediction <- ggplot(
  prediction_long,
  aes(x = Observed, y = Predicted)
) +
  geom_point(alpha = 0.65) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  facet_wrap(~ Model, scales = "free") +
  labs(
    title = "Observed versus predicted strength",
    x = "Observed strength (MPa)",
    y = "Predicted strength (MPa)"
  ) +
  theme_minimal(base_size = 10)

ggsave(
  file.path(data_dir, "observed_vs_predicted.png"),
  p_prediction, width = 11, height = 8, dpi = 300
)

# -----------------------------
# 14. Random forest importance
# -----------------------------

importance_values <- randomForest::importance(fit_rf)

importance_table <- data.frame(
  Variable = rownames(importance_values),
  importance_values,
  row.names = NULL,
  check.names = FALSE
)

cat("\nRandom forest variable importance:\n")
print(importance_table)

write.csv(
  importance_table,
  file.path(data_dir, "random_forest_importance.csv"),
  row.names = FALSE
)

png(
  file.path(data_dir, "random_forest_importance.png"),
  width = 2200, height = 1800, res = 300
)
varImpPlot(fit_rf, main = "Random forest variable importance")
dev.off()

# -----------------------------
# 15. Exploratory response surface
# -----------------------------

# This surface varies fly ash and age while holding other
# constituents at their training-set medians. It is exploratory:
# independently varying ingredients may create infeasible mixtures.

reference_values <- lapply(
  train[, predictor_names, drop = FALSE],
  median
)

flyash_grid <- seq(
  min(train$flyash),
  max(train$flyash),
  length.out = 80
)

age_grid <- seq(
  min(train$age),
  max(train$age),
  length.out = 80
)

rsm_grid <- expand.grid(
  flyash = flyash_grid,
  age = age_grid
)

rsm_grid$cement <- reference_values$cement
rsm_grid$slag <- reference_values$slag
rsm_grid$water <- reference_values$water
rsm_grid$superplasticizer <- reference_values$superplasticizer
rsm_grid$coarse_aggregate <- reference_values$coarse_aggregate
rsm_grid$fine_aggregate <- reference_values$fine_aggregate

rsm_grid$predicted_strength <- predict(
  fit_rsm,
  newdata = rsm_grid
)

p_surface <- ggplot(
  rsm_grid,
  aes(
    x = flyash,
    y = age,
    z = predicted_strength
  )
) +
  geom_contour_filled() +
  labs(
    title = "Exploratory response surface",
    x = "Fly ash (kg/m^3)",
    y = "Curing age (days)",
    fill = "Predicted strength\n(MPa)"
  ) +
  theme_minimal(base_size = 11)

ggsave(
  file.path(data_dir, "response_surface_contour.png"),
  p_surface, width = 8, height = 6, dpi = 300
)

# Highest predicted point on this grid.
# This is NOT a validated engineering optimum.
best_grid_point <- rsm_grid[
  which.max(rsm_grid$predicted_strength),
]

cat("\nExploratory highest-prediction grid point:\n")
print(best_grid_point)

write.csv(
  best_grid_point,
  file.path(data_dir, "exploratory_grid_candidate.csv"),
  row.names = FALSE
)

# -----------------------------
# 16. Record software information
# -----------------------------

capture.output(
  sessionInfo(),
  file = file.path(data_dir, "session_info.txt")
)

cat(
  "\nAnalysis completed. Results and figures are saved in:\n",
  normalizePath(data_dir),
  "\n"
)