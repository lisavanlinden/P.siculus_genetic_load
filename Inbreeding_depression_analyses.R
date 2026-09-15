# ============================================================
# Performance analyses
# ============================================================

library(dplyr)
library(readxl)
library(MCMCglmm)
library(Matrix)


# ============================================================
# 1. Load and prepare data
# ============================================================

df_per <- read_excel("performance_summary.xlsx")

df_per <- df_per %>%
  mutate(
    Froh = percent_genome_in_roh / 100
  )

# Convert selected variables to numeric
numeric_variables <- c(
  "BF_max",
  "max_speed",
  "max_speed_alt",
  "sp_head_av",
  "sp_midp_av",
  "sp_tail_av",
  "sp_tot_length_av",
  "average_sperm_velocity",
  "distance_sum",
  "sperm_count",
  "sperm_count0",
  "max_ex_distance_cm",
  "max_ex_time_s",
  "pha_diff_left",
  "pha_diff_rightleft",
  "tail_length",
  "head_width",
  "head_length",
  "head_height",
  "haemogr_paras_count",
  "trials_to_success",
  "n_success",
  "mean_latency",
  "mean_errors"
)

df_per <- df_per %>%
  mutate(
    across(
      all_of(numeric_variables),
      as.numeric
    )
  )


# Log-transform variables where appropriate
df_per <- df_per %>%
  mutate(
    log_SVL = log(SVL),
    log_BF_max = log(BF_max),
    log_max_speed = log(max_speed),
    log_average_sperm_velocity = log(average_sperm_velocity),
    log_n_success = log(n_success),
    log_trials_to_success = log(trials_to_success),
    log_mean_errors = log(mean_errors),
    log_mean_latency = log(mean_latency)
  )


# Calculate head size
df_per <- df_per %>%
  mutate(
    head_size = (
      head_height * head_width * head_length
    )^(1 / 3)
  )


# ============================================================
# 2. Helper function for GRM preparation
# ============================================================

prepare_grm <- function(grm_file, fam_file) {
  
  grm <- as.matrix(
    read.table(
      grm_file,
      header = FALSE
    )
  )
  
  ids <- read.table(
    fam_file,
    stringsAsFactors = FALSE
  )$V2
  
  rownames(grm) <- ids
  colnames(grm) <- ids
  
  # Ensure that the GRM is positive definite
  grm_positive <- make.positive.definite(grm)
  
  # Invert GRM and convert to sparse matrix
  grm_inverse <- solve(grm_positive)
  
  grm_inverse_sparse <- as(
    grm_inverse,
    "dgCMatrix"
  )
  
  return(grm_inverse_sparse)
}


# ============================================================
# 3. Priors
# ============================================================

prior_noninf <- list(
  G = list(
    G1 = list(V = 1, nu = 0.002),
    G2 = list(V = 1, nu = 0.002)
  ),
  R = list(
    V = 1,
    nu = 0.002
  )
)


# ============================================================
# 4. Bite force
# ============================================================

df_BF <- df_per %>%
  filter(!is.na(log_BF_max))

grm_inv_BF <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_bf.cXX.txt",
  fam_file = "psiculus_inbreeding_bf.fam"
)


model_BF_Froh <- MCMCglmm(
  log_BF_max ~ Froh * type + head_size,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(
    individual_ID = grm_inv_BF
  ),
  prior = prior_noninf,
  data = df_BF,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_BF_Froh)

save_model_summary(model_BF_Froh)


# ============================================================
# 5. Sprint speed
# ============================================================

df_sprint <- df_per %>%
  filter(!is.na(log_max_speed))

grm_inv_sprint <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_sprint.cXX.txt",
  fam_file = "psiculus_inbreeding_sprint.fam"
)


model_sprint_Froh <- MCMCglmm(
  log_max_speed ~ Froh * type + log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(
    individual_ID = grm_inv_sprint
  ),
  prior = prior_noninf,
  data = df_sprint,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_sprint_Froh)

save_model_summary(model_sprint_Froh)


# ============================================================
# 6. Sperm concentration
# ============================================================

df_spermc <- df_per %>%
  filter(!is.na(sperm_count))

grm_inv_spermc <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_spermc.cXX.txt",
  fam_file = "psiculus_inbreeding_spermc.fam"
)


model_spermc_Froh <- MCMCglmm(
  sperm_count ~ Froh * type + log_SVL,
  random = ~ individual_ID + population,
  family = "poisson",
  ginverse = list(
    individual_ID = grm_inv_spermc
  ),
  prior = prior_noninf,
  data = df_spermc,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_spermc_Froh)

save_model_summary(model_spermc_Froh)


# ============================================================
# 7. Sperm length
# ============================================================

df_sperml <- df_per %>%
  filter(
    !is.na(sp_tot_length_av),
    !is.na(sperm_count)
  )

grm_inv_sperml <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_sperml.cXX.txt",
  fam_file = "psiculus_inbreeding_sperml.fam"
)


model_sperml_Froh <- MCMCglmm(
  sp_tot_length_av ~ Froh * type + sperm_count + log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(
    individual_ID = grm_inv_sperml
  ),
  prior = prior_noninf,
  data = df_sperml,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_sperml_Froh)

save_model_summary(model_sperml_Froh)


# ============================================================
# 8. Sperm velocity
# ============================================================

df_spermv <- df_per %>%
  filter(!is.na(log_average_sperm_velocity))

grm_inv_spermv <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_spermv.cXX.txt",
  fam_file = "psiculus_inbreeding_spermv.fam"
)


model_spermv_Froh <- MCMCglmm(
  log_average_sperm_velocity ~
    Froh * type +
    sp_tot_length_av +
    log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(
    individual_ID = grm_inv_spermv
  ),
  prior = prior_noninf,
  data = df_spermv,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_spermv_Froh)

save_model_summary(model_spermv_Froh)


# ============================================================
# 9. Immune function
# ============================================================

df_pha <- df_per %>%
  filter(
    !is.na(pha_diff_rightleft),
    !is.na(haemogr_paras_count)
  )

grm_inv_pha <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_pha.cXX.txt",
  fam_file = "psiculus_inbreeding_pha.fam"
)


model_pha_Froh <- MCMCglmm(
  pha_diff_rightleft ~
    Froh * type +
    haemogr_paras_count +
    log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(
    individual_ID = grm_inv_pha
  ),
  prior = prior_noninf,
  data = df_pha,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_pha_Froh)

save_model_summary(model_pha_Froh)


# ============================================================
# 10. Spatial learning
# ============================================================

# Correlation between number of successes and trials to success
cor.test(
  df_per$n_success,
  df_per$trials_to_success,
  use = "complete.obs"
)


# Prepare cognitive dataset
df_cog <- df_per %>%
  filter(!is.na(n_success))

grm_inv_cog <- prepare_grm(
  grm_file = "GRM_psiculus_inbreeding_cog.cXX.txt",
  fam_file = "psiculus_inbreeding_cog.fam"
)


# ------------------------------------------------------------
# Trials to success
# ------------------------------------------------------------

model_cog_Froh <- MCMCglmm(
  trials_to_success ~ Froh * type + log_SVL,
  random = ~ individual_ID + population,
  family = "poisson",
  ginverse = list(
    individual_ID = grm_inv_cog
  ),
  prior = prior_noninf,
  data = df_cog,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_cog_Froh)

save_model_summary(model_cog_Froh)


# ============================================================
# STANDARDIZED MODELS
# ============================================================

# Continuous response variables and continuous predictors were
# standardized to mean 0 and standard deviation 1.
#
# For Gaussian models, coefficients represent the expected change
# in the response, expressed in standard deviations, associated
# with a one-standard-deviation increase in the predictor.
#
# For Poisson models, continuous predictors were standardized.
# Their coefficients represent the effect of a one-standard-deviation
# increase in the predictor on the log scale of the response.
#
# These models are useful for comparing effect sizes across traits.
# ============================================================


# ------------------------------------------------------------
# Bite force
# ------------------------------------------------------------

df_BF <- df_BF %>%
  mutate(
    z_log_BF_max = as.numeric(scale(log_BF_max)),
    z_Froh = as.numeric(scale(Froh)),
    z_head_size = as.numeric(scale(head_size))
  )

model_GRM_BF_std <- MCMCglmm(
  z_log_BF_max ~ z_Froh * type + z_head_size,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_BF,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_BF_std)
save_model_summary(model_GRM_BF_std)


# ------------------------------------------------------------
# Sprint speed
# ------------------------------------------------------------

df_sprint <- df_sprint %>%
  mutate(
    z_log_max_speed = as.numeric(scale(log_max_speed)),
    z_Froh = as.numeric(scale(Froh)),
    z_log_SVL = as.numeric(scale(log_SVL))
  )

model_GRM_SPRINT_std <- MCMCglmm(
  z_log_max_speed ~ z_Froh * type + z_log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_sprint,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_SPRINT_std)
save_model_summary(model_GRM_SPRINT_std)


# ------------------------------------------------------------
# Sperm count
# ------------------------------------------------------------

# The response remains unstandardized because it is modelled
# using a Poisson distribution.

df_spermc <- df_spermc %>%
  mutate(
    z_Froh = as.numeric(scale(Froh)),
    z_log_SVL = as.numeric(scale(log_SVL))
  )

model_GRM_SPERMC_std <- MCMCglmm(
  sperm_count ~ z_Froh * type + z_log_SVL,
  random = ~ individual_ID + population,
  family = "poisson",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_spermc,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_SPERMC_std)
save_model_summary(model_GRM_SPERMC_std)


# ------------------------------------------------------------
# Sperm length
# ------------------------------------------------------------

df_sperml <- df_sperml %>%
  mutate(
    z_sp_tot_length_av = as.numeric(scale(sp_tot_length_av)),
    z_Froh = as.numeric(scale(Froh)),
    z_log_SVL = as.numeric(scale(log_SVL))
  )

model_GRM_SPERML_std <- MCMCglmm(
  z_sp_tot_length_av ~ z_Froh * type + z_log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_sperml,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_SPERML_std)
save_model_summary(model_GRM_SPERML_std)


# ------------------------------------------------------------
# Sperm velocity
# ------------------------------------------------------------

df_spermv <- df_spermv %>%
  mutate(
    z_log_average_sperm_velocity = as.numeric(
      scale(log_average_sperm_velocity)
    ),
    z_sp_tot_length_av = as.numeric(scale(sp_tot_length_av)),
    z_Froh = as.numeric(scale(Froh)),
    z_log_SVL = as.numeric(scale(log_SVL))
  )

model_GRM_SPERMV_std <- MCMCglmm(
  z_log_average_sperm_velocity ~
    z_Froh * type +
    z_sp_tot_length_av +
    z_log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_spermv,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_SPERMV_std)
save_model_summary(model_GRM_SPERMV_std)


# ------------------------------------------------------------
# Immune function
# ------------------------------------------------------------

df_pha3 <- df_pha3 %>%
  mutate(
    z_pha_diff_rightleft = as.numeric(scale(pha_diff_rightleft)),
    z_haemogr_paras_count = as.numeric(scale(haemogr_paras_count)),
    z_Froh = as.numeric(scale(Froh)),
    z_log_SVL = as.numeric(scale(log_SVL))
  )

model_GRM_PHA_std <- MCMCglmm(
  z_pha_diff_rightleft ~
    z_Froh * type +
    z_haemogr_paras_count +
    z_log_SVL,
  random = ~ individual_ID + population,
  family = "gaussian",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_pha3,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_PHA_std)
save_model_summary(model_GRM_PHA_std)


# ------------------------------------------------------------
# Spatial cognition
# ------------------------------------------------------------

# The response remains unstandardized because it is modelled
# using a Poisson distribution.

df_cog <- df_cog %>%
  mutate(
    z_Froh = as.numeric(scale(Froh)),
    z_log_SVL = as.numeric(scale(log_SVL))
  )

model_GRM_COG_std <- MCMCglmm(
  trials_to_success ~ z_Froh * type + z_log_SVL,
  random = ~ individual_ID + population,
  family = "poisson",
  ginverse = list(individual_ID = grm_inv_sp),
  prior = prior_noninf,
  data = df_cog,
  nitt = 2000000,
  burnin = 2000,
  thin = 200
)

summary(model_GRM_COG_std)
save_model_summary(model_GRM_COG_std)

# ============================================================
# EXTRACT POSTERIOR SAMPLES AND GENERATE PREDICTIONS
# ============================================================

# ------------------------------------------------------------
# Extract posterior samples for fixed effects
# ------------------------------------------------------------

posterior_samples <- as.data.frame(model_GRM_BF2$Sol)

posterior_fixed_effects <- posterior_samples[, c(
  "(Intercept)",
  "Froh",
  "typemainland",
  "Froh:typemainland",
  "head_size"
)]

write.table(
  posterior_fixed_effects,
  file = "mcmcglmm_posterior_fixed_effects.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


# ------------------------------------------------------------
# Generate predictions across the observed Froh range
# ------------------------------------------------------------

Froh_seq <- seq(
  from = min(df_BF$Froh, na.rm = TRUE),
  to = max(df_BF$Froh, na.rm = TRUE),
  length.out = 100
)

mean_head_size <- mean(df_BF$head_size, na.rm = TRUE)


predict_bite_force <- function(type) {
  
  if (type == "island") {
    
    linear_predictor <- outer(
      posterior_samples$Froh,
      Froh_seq,
      FUN = "*"
    ) +
      posterior_samples$`(Intercept)` +
      posterior_samples$head_size * mean_head_size
    
  } else if (type == "mainland") {
    
    linear_predictor <- outer(
      posterior_samples$Froh +
        posterior_samples$`Froh:typemainland`,
      Froh_seq,
      FUN = "*"
    ) +
      posterior_samples$`(Intercept)` +
      posterior_samples$typemainland +
      posterior_samples$head_size * mean_head_size
    
  } else {
    stop("type must be either 'island' or 'mainland'.")
  }
  
  predicted_bite_force <- exp(linear_predictor)
  
  tibble(
    Froh = Froh_seq,
    BF_mean = colMeans(predicted_bite_force),
    BF_low = apply(
      predicted_bite_force,
      2,
      quantile,
      probs = 0.025
    ),
    BF_high = apply(
      predicted_bite_force,
      2,
      quantile,
      probs = 0.975
    ),
    type = type
  )
}


predictions_BF <- bind_rows(
  predict_bite_force("island"),
  predict_bite_force("mainland")
)

write.table(
  predictions_BF,
  file = "mcmcglmm_BF_Froh_predictions.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


# ------------------------------------------------------------
# Export raw data used for the bite-force analysis
# ------------------------------------------------------------

df_BF_model <- df_BF %>%
  select(
    log_BF_max,
    Froh,
    head_size,
    type
  )

summary(df_BF_model)
table(df_BF_model$type)

write.table(
  df_BF_model,
  file = "mcmcglmm_raw_data_BF_Froh.tsv",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)