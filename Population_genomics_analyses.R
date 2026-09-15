# ============================================================
# 1. Heterozygosity
# ============================================================

library(dplyr)
library(tidyr)
library(readxl)
library(lme4)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df_het <- read_excel("ANGSD_summary.xlsx")

head(df_het)
summary(df_het)


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

# Heterozygosity for population BJ
het_BJ_summary <- df_het %>%
  filter(population == "BJ") %>%
  summarise(
    mean_het = mean(mean_100KB_het, na.rm = TRUE),
    sd_het = sd(mean_100KB_het, na.rm = TRUE),
    n = sum(!is.na(mean_100KB_het))
  )

het_BJ_summary


# Heterozygosity by region
het_summary_region <- df_het %>%
  group_by(region) %>%
  summarise(
    mean_het = mean(mean_100KB_het, na.rm = TRUE),
    sd_het = sd(mean_100KB_het, na.rm = TRUE),
    se_het = sd_het / sqrt(sum(!is.na(mean_100KB_het))),
    n = sum(!is.na(mean_100KB_het)),
    .groups = "drop"
  )

het_summary_region


# Heterozygosity by population
het_summary_population <- df_het %>%
  group_by(population) %>%
  summarise(
    mean_het = mean(mean_100KB_het, na.rm = TRUE),
    sd_het = sd(mean_100KB_het, na.rm = TRUE),
    se_het = sd_het / sqrt(sum(!is.na(mean_100KB_het))),
    n = sum(!is.na(mean_100KB_het)),
    .groups = "drop"
  )

het_summary_population


# ------------------------------------------------------------
# Mixed-effects model: heterozygosity
# ------------------------------------------------------------

model_het <- lmer(
  mean_100KB_het ~ region + (1 | population),
  data = df_het
)

summary(model_het)


# ============================================================
# 2. Runs of homozygosity
# ============================================================

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df_roh <- read_excel("roh_bcftools_summary.xlsx")

head(df_roh)
summary(df_roh)


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

# Percentage of genome in ROHs by region
roh_summary_region <- df_roh %>%
  group_by(Region) %>%
  summarise(
    mean_roh_perc = mean(percent_genome_in_roh, na.rm = TRUE),
    sd_roh_perc = sd(percent_genome_in_roh, na.rm = TRUE),
    se_roh_perc = sd_roh_perc / sqrt(sum(!is.na(percent_genome_in_roh))),
    n = sum(!is.na(percent_genome_in_roh)),
    .groups = "drop"
  )

roh_summary_region


# Percentage of genome in ROHs by population
roh_summary_population <- df_roh %>%
  group_by(Population) %>%
  summarise(
    mean_roh_perc = mean(percent_genome_in_roh, na.rm = TRUE),
    sd_roh_perc = sd(percent_genome_in_roh, na.rm = TRUE),
    se_roh_perc = sd_roh_perc / sqrt(sum(!is.na(percent_genome_in_roh))),
    n = sum(!is.na(percent_genome_in_roh)),
    .groups = "drop"
  )

roh_summary_population


# ------------------------------------------------------------
# Mixed-effects model: percentage of genome in ROHs
# ------------------------------------------------------------

model_roh <- lmer(
  percent_genome_in_roh ~ Region + (1 | Population),
  data = df_roh
)

summary(model_roh)


# ------------------------------------------------------------
# Binomial model: proportion of genome in ROHs
# ------------------------------------------------------------

genome_size <- 1423562263  # bp

model_roh_binom <- glmer(
  cbind(
    total_roh_bp,
    genome_size - total_roh_bp
  ) ~ Region + (1 | Population),
  data = df_roh,
  family = binomial(link = "logit")
)

summary(model_roh_binom)


# ============================================================
# 3. Association between FROH and heterozygosity
# ============================================================

# ------------------------------------------------------------
# Combine heterozygosity and ROH data
# ------------------------------------------------------------

df_combined <- df_het %>%
  select(
    individual,
    region,
    population,
    mean_100KB_het
  ) %>%
  left_join(
    df_roh %>%
      select(
        Sample,
        percent_genome_in_roh
      ),
    by = c("individual" = "Sample")
  ) %>%
  mutate(
    Froh = percent_genome_in_roh / 100
  )


head(df_combined)
summary(df_combined)


# ------------------------------------------------------------
# Mixed-effects model: FROH and heterozygosity
# ------------------------------------------------------------

model_froh_het <- lmer(
  mean_100KB_het ~ Froh * region + (1 | population),
  data = df_combined
)

summary(model_froh_het)