# ============================================================
# Genetic load analyses
# ============================================================

library(dplyr)
library(tidyr)
library(readxl)
library(lme4)


# ============================================================
# 1. Genetic load: DEL/SYN and LOF/SYN ratios
# ============================================================

df_snpeff <- read_excel("snpeff_counts.xlsx")

df_ratio <- df_snpeff %>%
  select(
    sample,
    population,
    population_type,
    category,
    Hom,
    Het
  ) %>%
  pivot_wider(
    names_from = category,
    values_from = c(Hom, Het)
  ) %>%
  mutate(
    Hom_DEL_SYN = Hom_DEL / Hom_SYN,
    Het_DEL_SYN = Het_DEL / Het_SYN,
    Hom_LOF_SYN = Hom_LOF / Hom_SYN,
    Het_LOF_SYN = Het_LOF / Het_SYN
  )


# Mixed-effects models testing differences between island
# and mainland populations, with population as a random effect

model_hom_del_syn <- lmer(
  log(Hom_DEL_SYN) ~ population_type + (1 | population),
  data = df_ratio
)

model_het_del_syn <- lmer(
  log(Het_DEL_SYN) ~ population_type + (1 | population),
  data = df_ratio
)

model_hom_lof_syn <- lmer(
  log(Hom_LOF_SYN) ~ population_type + (1 | population),
  data = df_ratio
)

model_het_lof_syn <- lmer(
  log(Het_LOF_SYN) ~ population_type + (1 | population),
  data = df_ratio
)


summary(model_hom_del_syn)
summary(model_het_del_syn)
summary(model_hom_lof_syn)
summary(model_het_lof_syn)


# ============================================================
# 2. Association between homozygous variants and FROH
# ============================================================

df_counts <- df_snpeff %>%
  select(sample, category, Hom) %>%
  pivot_wider(
    names_from = category,
    values_from = Hom
  ) %>%
  rename(
    hom_del = DEL,
    hom_lof = LOF,
    hom_syn = SYN
  )


df_roh <- read_excel("roh_bcftools_summary.xlsx")

df_froh <- df_roh %>%
  rename(sample = Sample) %>%
  select(
    sample,
    Froh,
    Region,
    Population
  ) %>%
  left_join(df_counts, by = "sample") %>%
  mutate(
    log_hom_del = log(hom_del),
    log_hom_lof = log(hom_lof),
    log_hom_syn = log(hom_syn)
  )



df_froh <- df_froh %>%
  group_by(Region) %>%
  mutate(
    Froh_centered = Froh - mean(Froh, na.rm = TRUE)
  ) %>%
  ungroup()


model_del_centered <- lmer(
  log_hom_del ~ Froh_centered * Region + log_hom_syn +
    (1 | Population),
  data = df_froh
)

model_lof_centered <- lmer(
  log_hom_lof ~ Froh_centered * Region + log_hom_syn +
    (1 | Population),
  data = df_froh
)


summary(model_del_centered)
summary(model_lof_centered)


# ============================================================
# 4. Genetic load inside versus outside ROHs
# ============================================================

df_rohload <- read_excel("roh_load_counts.xlsx")

df_rohload_ratio <- df_rohload %>%
  mutate(
    DEL_SYN = total_del / total_syn,
    LOF_SYN = total_lof / total_syn
  ) %>%
  rename(
    ROH_location = location
  )


# ------------------------------------------------------------
# 4a. Mainland only
# ------------------------------------------------------------

df_mainland <- df_rohload_ratio %>%
  filter(region == "mainland")

model_mainland_del <- lmer(
  log(DEL_SYN) ~ ROH_location + (1 | population),
  data = df_mainland
)

model_mainland_lof <- lmer(
  log(LOF_SYN) ~ ROH_location + (1 | population),
  data = df_mainland
)


summary(model_mainland_del)
summary(model_mainland_lof)


# ------------------------------------------------------------
# 4b. Island only
# ------------------------------------------------------------

df_island <- df_rohload_ratio %>%
  filter(region == "island")

model_island_del <- lmer(
  log(DEL_SYN) ~ ROH_location + (1 | population),
  data = df_island
)

model_island_lof <- lmer(
  log(LOF_SYN) ~ ROH_location + (1 | population),
  data = df_island
)


summary(model_island_del)
summary(model_island_lof)


# ------------------------------------------------------------
# 4c. Combined model testing the region interaction
# ------------------------------------------------------------

model_roh_del <- lmer(
  log(DEL_SYN) ~ ROH_location * region + (1 | population),
  data = df_rohload_ratio
)

model_roh_lof <- lmer(
  log(LOF_SYN) ~ ROH_location * region + (1 | population),
  data = df_rohload_ratio
)

summary(model_roh_del)
summary(model_roh_lof)