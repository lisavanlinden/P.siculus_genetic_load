# Analyses and plots population genomics manuscript


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import os
import gzip
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo
from skbio import DistanceMatrix
from skbio.tree import nj
from io import StringIO
from skbio import TreeNode
jn = os.path.join
import sys, ete3
sys.path.append('/data/antwerpen/grp/asvardal/hs_tools')
from pypopgen3.modules import treetools as tt
from pypopgen3.modules import diversity
from ete3 import TreeStyle, NodeStyle
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import matplotlib.font_manager as fm
import pysam
import math
import glob
```


```python
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
```


```python

```

## 11. ROH-GWAS WINDOWS


```python
def window_midpoint(window_id):
    """
    Extract midpoint from WINDOW_ID string:
    e.g. 'OZ076838.1:33015-132508'
    """
    try:
        coords = window_id.split(":")[1]
        start, end = coords.split("-")
        return (int(start) + int(end)) // 2
    except Exception:
        return np.nan

def chrom_numeric(chrom):
    """
    Extract numeric chromosome index from names like:
    'OZ076838.1' -> 76838
    """
    try:
        m = re.search(r"OZ(\d+)", chrom)
        return int(m.group(1)) if m else chrom
    except Exception:
        return chrom

def plot_manhattan(files, title, neff_genome):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ----------------------------
    # Load and concatenate
    # ----------------------------
    df = pd.concat(
        (pd.read_csv(f, sep="\t") for f in files),
        ignore_index=True
    )

    # ----------------------------
    # Compute window midpoint (visual only)
    # ----------------------------
    df["POS"] = df["WINDOW_ID"].apply(window_midpoint)
    df["LOGP"] = -np.log10(df["P_ROH_STATUS"])

    # ----------------------------
    # Chromosome order
    # ----------------------------
    chrom_order = sorted(
        df["CHR"].unique(),
        key=chrom_numeric
    )

    # ----------------------------
    # Cumulative positions
    # ----------------------------
    chrom_offsets = {}
    current_offset = 0

    for chrom in chrom_order:
        chrom_offsets[chrom] = current_offset
        current_offset += df.loc[df["CHR"] == chrom, "POS"].max()

    df["POS_CUM"] = df.apply(
        lambda r: r["POS"] + chrom_offsets[r["CHR"]],
        axis=1
    )

    # ----------------------------
    # Chromosome midpoints for x-axis labels
    # ----------------------------
    chrom_centers = []
    chrom_labels = []

    for chrom in chrom_order:
        sub = df[df["CHR"] == chrom]
        center = (sub["POS_CUM"].min() + sub["POS_CUM"].max()) / 2
        chrom_centers.append(center)
        chrom_labels.append(chrom)


    # ----------------------------
    # Classical Bonferroni
    # ----------------------------
    valid = df["P_ROH_STATUS"].notna() & np.isfinite(df["P_ROH_STATUS"])
    n_tests = valid.sum()

    bonf_p = 0.05 / n_tests
    bonf_logp = -np.log10(bonf_p)

    # ----------------------------
    # Effective Bonferroni (simpleM)
    # ----------------------------
    eff_bonf_p = 0.05 / neff_genome
    eff_bonf_logp = -np.log10(eff_bonf_p)

        # ----------------------------
    # Summary statistics for printing
    # ----------------------------
    valid = df["P_ROH_STATUS"].notna() & np.isfinite(df["P_ROH_STATUS"])

    n_windows_total = len(df)
    n_windows_tested = valid.sum()
    n_windows_na = n_windows_total - n_windows_tested

    # Classical Bonferroni
    bonf_p = 0.05 / n_windows_tested
    bonf_logp = -np.log10(bonf_p)

    # Effective Bonferroni
    eff_bonf_p = 0.05 / neff_genome
    eff_bonf_logp = -np.log10(eff_bonf_p)

    # Chromosome-level stats
    windows_per_chr = (
        df.loc[valid]
        .groupby("CHR")
        .size()
    )

    n_chromosomes = windows_per_chr.shape[0]

    # Sample size (if IID exists)
    if "IID" in df.columns:
        sample_size = df["IID"].nunique()
    else:
        sample_size = "NA"

    # ----------------------------
    # Print nicely formatted block
    # ----------------------------
    print("\n" + "=" * 60)
    print("ROH-GWAS Manhattan plot summary")
    print("=" * 60)
    print(f"Total windows loaded       : {n_windows_total:,}")
    print(f"Windows tested             : {n_windows_tested:,}")
    print(f"Windows excluded (NA/inf)  : {n_windows_na:,}")
    print()
    print(f"Number of chromosomes      : {n_chromosomes}")
    print(f"Windows per chromosome     :")
    print(f"  mean   = {windows_per_chr.mean():.1f}")
    print(f"  median = {windows_per_chr.median():.1f}")
    print(f"  range  = {windows_per_chr.min()}–{windows_per_chr.max()}")
    print()
    print(f"Sample size (N individuals): {sample_size}")
    print()
    print("Multiple testing thresholds")
    print(f"  Classical Bonferroni p   : {bonf_p:.3e}")
    print(f"  Classical Bonferroni -log10(p): {bonf_logp:.2f}")
    print(f"  Effective Bonferroni p   : {eff_bonf_p:.3e}")
    print(f"  Effective Bonferroni -log10(p): {eff_bonf_logp:.2f}")
    print("=" * 60 + "\n")


    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(14, 5))
    colors = ["#4C72B0", "#DD8452"]
    
    for i, chrom in enumerate(chrom_order):
        sub = df[df["CHR"] == chrom]
        plt.scatter(
            sub["POS_CUM"],
            sub["LOGP"],
            s=10,
            color=colors[i % 2]
        )
    
    # Classical Bonferroni
    plt.axhline(
        bonf_logp,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Classical Bonferroni"
    )
    
    # Effective Bonferroni
    plt.axhline(
        eff_bonf_logp,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Effective Bonferroni (Nₑff)"
    )
    
    # --- Chromosome labels instead of positions ---
    plt.xticks(chrom_centers, chrom_labels, rotation=90)
    
    plt.xlabel("Chromosome")
    plt.ylabel(r"$-\log_{10}(P)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


```


```python
def plot_qq(files, title):
    """
    QQ plot for ROH-GWAS window-based p-values.

    Parameters
    ----------
    files : list of str
        List of GWAS result files (e.g. roh_gwas_windows_ALL_*.tsv)
    title : str
        Plot title
    """

    # ----------------------------
    # Load and concatenate
    # ----------------------------
    df = pd.concat(
        (pd.read_csv(f, sep="\t") for f in files),
        ignore_index=True
    )

    # ----------------------------
    # Extract and clean p-values
    # ----------------------------
    pvals = df["P_ROH_STATUS"].dropna().values
    pvals = pvals[pvals > 0]   # avoid log(0)

    n = len(pvals)
    if n == 0:
        print("No valid p-values to plot.")
        return

    # ----------------------------
    # Expected vs observed
    # ----------------------------
    pvals_sorted = np.sort(pvals)
    exp = -np.log10(np.arange(1, n + 1) / (n + 1))
    obs = -np.log10(pvals_sorted)

    # ----------------------------
    # Genomic inflation factor (λ)
    # ----------------------------
    chisq = np.quantile(
        -2 * np.log(pvals_sorted),
        0.5
    )
    lambda_gc = chisq / 0.4549364  # median of chi^2_1

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(5, 5))
    plt.scatter(exp, obs, s=10, alpha=0.7)
    plt.plot([0, exp.max()], [0, exp.max()], color="red", lw=1)

    plt.xlabel("Expected $-\\log_{10}(P)$")
    plt.ylabel("Observed $-\\log_{10}(P)$")
    plt.title(f"{title}\nλ = {lambda_gc:.3f}")

    plt.tight_layout()
    plt.show()

```

### 11.1. 1500 SNPs - 50% threshold

#### 11.1.1. Mainland

##### SimpleM 


```python
POP_PREFIXES = {
    "ML": ("VS", "24VS", "TM", "24TM", "RG", "CV", "FV"),   # VS belongs to ML
    "IL": ("SC","24SC", "PK", "24PK", "PM", "24PM", "KP")         # only IL
}
```


```python
def subset_population(df, pop):
    """
    Subset dataframe based on IID prefixes belonging to a population.
    """
    if "IID" not in df.columns:
        raise ValueError("IID column not found in input dataframe")

    if pop not in POP_PREFIXES:
        raise ValueError(f"Unknown population '{pop}'. Valid: {list(POP_PREFIXES)}")

    prefixes = POP_PREFIXES[pop]

    mask = df["IID"].astype(str).str.startswith(prefixes)
    return df[mask].copy()

```


```python
import pandas as pd
import numpy as np
from numpy.linalg import eigh
def neff_from_roh_matrix(X, variance_threshold=0.995):
    """
    Compute effective number of independent tests (simpleM-style)
    from an individual × window ROH matrix.
    """

    # Drop windows with zero variance (all 0 or all 1)
    variances = X.var(axis=0)
    X = X.loc[:, variances > 0]

    # If too few windows remain, N_eff = number of windows
    if X.shape[1] <= 1:
        return X.shape[1]

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    X_std = X_std.fillna(0)

    # Correlation matrix
    R = np.corrcoef(X_std.values, rowvar=False)

    # Eigenvalues only (backward-compatible)
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(eigvals)[::-1]

    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    neff = np.searchsorted(cumvar, variance_threshold) + 1

    return neff


def compute_neff_from_window_folder(
    input_dir,
    pop,
    variance_threshold=0.995,
    file_pattern="gwas_input_windows_covariates_*.tsv"
):
    """
    Compute genome-wide N_eff for windowed ROH-GWAS by scanning
    per-chromosome input files in a folder.

    Parameters
    ----------
    input_dir : str
        Directory containing per-chromosome window input files
    pop : str
        'ML' or 'IL'
    variance_threshold : float
        Cumulative variance threshold (default 0.995, simpleM)
    file_pattern : str
        Glob pattern to match chromosome files

    Returns
    -------
    total_neff : int
        Genome-wide effective number of independent windows
    summary_df : pandas.DataFrame
        Per-chromosome N_windows and N_eff
    """

    files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))

    if not files:
        raise FileNotFoundError("No input files found in directory")

    total_neff = 0
    summary = []

    print(f"\nComputing N_eff for {pop}")
    print(f"Scanning {len(files)} chromosome files\n")

    for f in files:
        df = pd.read_csv(f, sep="\t")

        # Population subset
        df = subset_population(df, pop)

        if df.empty:
            continue

        chrom = df["CHR"].iloc[0]

        # Individual × window matrix
        X = (
            df.pivot_table(
                index="IID",
                columns="WINDOW_ID",
                values="ROH_STATUS",
                fill_value=0
            )
            .astype(float)
        )

        if X.shape[1] < 2:
            continue

        neff_chr = neff_from_roh_matrix(X, variance_threshold)

        total_neff += neff_chr

        summary.append({
            "CHR": chrom,
            "N_windows": X.shape[1],
            "N_eff": neff_chr
        })

        print(f"{pop} {chrom}: windows={X.shape[1]}  N_eff={neff_chr}")

    print(f"\n{pop} TOTAL genome-wide N_eff = {total_neff}\n")

    return total_neff, pd.DataFrame(summary)


```


```python
INPUT_DIR = "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/roh_gwas_input_windows/1500_50"

neff_ml, neff_ml_chr = compute_neff_from_window_folder(
    INPUT_DIR,
    pop="ML"
)

neff_il, neff_il_chr = compute_neff_from_window_folder(
    INPUT_DIR,
    pop="IL"
)

```

##### Exploratory plots


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_MAINLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (MAINLAND) – 1500 SNPs, 50% threshold",
    neff_genome=947
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_50/roh_gwas_windows_MAINLAND_*.tsv"),
    "ROH-GWAS QQ plot (MAINLAND) - 1500 snps, 50% threshold"
)

```

##### Significant windows


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "gwas_roh/roh_gwas_results_windows/1500_50/"
)

# Choose group: ALL, MAINLAND, ISLAND
GROUP = "MAINLAND"

files = sorted(
    glob.glob(f"{RESULTS_DIR}/roh_gwas_windows_{GROUP}_*.tsv")
)

print(f"Found {len(files)} result files")

# ----------------------------
# Load all result files
# ----------------------------
df_ML = pd.concat(
    (pd.read_csv(f, sep="\t") for f in files),
    ignore_index=True
)

print(f"Total windows loaded (raw): {df_ML.shape[0]}")

# Load data
df_ML = df_ML.loc[
    df_ML["P_ROH_STATUS"].notna()
    & np.isfinite(df_ML["P_ROH_STATUS"])
].copy()

# Compute -log10(p) exactly as in the plot
df_ML["LOGP"] = -np.log10(df_ML["P_ROH_STATUS"])
print(f"Total windows loaded: {df_ML.shape[0]}")

neff_genome_ML = 947  # your simpleM result

p_eff_ML = 0.05 / neff_genome_ML
logp_eff_ML = -np.log10(p_eff_ML)

print(f"Neff Bonferroni threshold:")
print(f"  p ≤ {p_eff_ML:.3e}")
print(f"  -log10(p) ≥ {logp_eff_ML:.4f}")

sig_df_ML = df_ML.loc[
    df_ML["LOGP"] >= logp_eff_ML
].copy()

print(f"Significant windows (Neff, LOGP-based): {sig_df_ML.shape[0]}")
print(sig_df_ML)
```


```python
print(df_ML)
```


```python
df_ML["BETA_ROH_STATUS"].describe()
```


```python
is_sig = df_ML["LOGP"] >= logp_eff_ML
beta_min = -3
beta_max = 3

beta_ok = (
    df_ML["BETA_ROH_STATUS"].notna()
    & (df_ML["BETA_ROH_STATUS"] >= beta_min)
    & (df_ML["BETA_ROH_STATUS"] <= beta_max)
)
plot_mask = beta_ok
sig_plot_mask = beta_ok & is_sig

n_excluded = (~beta_ok).sum()
print(f"Effect sizes excluded from plot (|β| outside [{beta_min}, {beta_max}]): {n_excluded}")


plt.figure(figsize=(10, 4))

# Non-significant (within beta limits)
plt.scatter(
    df_ML.loc[plot_mask & ~is_sig].index,
    df_ML.loc[plot_mask & ~is_sig, "BETA_ROH_STATUS"],
    color="lightgrey",
    s=8,
    alpha=0.7,
    label="Not significant"
)

# Significant (within beta limits)
plt.scatter(
    df_ML.loc[sig_plot_mask].index,
    df_ML.loc[sig_plot_mask, "BETA_ROH_STATUS"],
    color="red",
    s=15,
    label="Neff significant"
)

plt.axhline(0, color="black", linewidth=0.8)

plt.ylim(beta_min, beta_max)

plt.xlabel("Window index")
plt.ylabel("Effect size (β_ROH_STATUS)")
plt.title("ROH-GWAS effect sizes (Mainland)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

```


```python
INPUT_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "roh_gwas_input_windows/1500_50/"
)

input_files = sorted(
    glob.glob(f"{INPUT_DIR}/gwas_input_windows_covariates_*.tsv")
)

df_input = pd.concat(
    (pd.read_csv(f, sep="\t") for f in input_files),
    ignore_index=True
)

print(f"Raw input rows loaded: {df_input.shape[0]}")

df_input_ML = df_input[df_input["PC1_ML"].notna()].copy()
df_input_ML["Population"] = df_input_ML["IID"].str.extract(r"([A-Z]+)")
df_input_ML["Population"] = df_input_ML["Population"].str.replace(r"^24", "", regex=True)

sig_windows_ML = sig_df_ML["WINDOW_ID"].unique()
print(f"Significant windows: {len(sig_windows_ML)}")

df_plot_ML = df_input_ML[df_input_ML["WINDOW_ID"].isin(sig_windows_ML)].copy()

color_map = {
    'PJ':'#6BAED6', 'MP': '#2171B5', 'KP': '#4292C6', 'PM': '#08306B',
    'BJ':'#08519C', 'SC':'#9ECAE1', 'PK':'#C6DBEF', 'VS':'#238B45',
    'TM':'#00441B', 'RG':'#BAE4B3', 'SP':'#74C476', 'TR':'#006D2C',
    'PL':'#41AB5D'
}

windows_ML = df_plot_ML["WINDOW_ID"].unique()
print(f"Plotting {len(windows_ML)} significant windows")

for window in windows_ML:

    df_w = df_plot_ML[df_plot_ML["WINDOW_ID"] == window]

    # Skip windows with no ROH variation
    if df_w["ROH_STATUS"].nunique() < 2:
        continue

    # Get effect size from GWAS results
    beta = sig_df_ML.loc[
        sig_df_ML["WINDOW_ID"] == window,
        "BETA_ROH_STATUS"
    ].iloc[0]

    plt.figure(figsize=(6, 4))

    # Boxplot
    sns.boxplot(
        x="ROH_STATUS",
        y="PHENO_z",     # or PHENO if you prefer raw scale
        data=df_w,
        showcaps=True,
        boxprops={"facecolor": "none"},
        width=0.5
    )

    # Population-coloured points
    sns.stripplot(
        x="ROH_STATUS",
        y="PHENO_z",
        data=df_w,
        hue="Population",
        dodge=True,
        jitter=True,
        size=6,
        alpha=0.8,
        palette=color_map
    )

    plt.xlabel("ROH status (0 = non-ROH, 1 = ROH)")
    plt.ylabel("Phenotype (z)")
    plt.title(f"{window}\nROH effect β = {beta:.3f}")

    plt.legend(
        title="Population",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

```


```python
sig_df_ML[["WINDOW_ID", "CHR", "BETA_ROH_STATUS", "N_ROH", "N_NONROH", "P_ROH_STATUS", "LOGP"]] \
    .to_csv("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_50/significant_windows/roh_neff_significant_windows_ML.tsv", sep="\t", index=False)

```


```python
sig_df_ML.head()
```


```python

```

#### 11.1.2. Island

##### Exploratory plots


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_ISLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (ISLAND) – 1500 SNPs, 50% threshold",
    neff_genome=1409
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_50/roh_gwas_windows_ISLAND_*.tsv"),
    "ROH-GWAS QQ plot (ISLAND) - 1500 snps, 50% threshold"
)

```

##### Significant snps


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "gwas_roh/roh_gwas_results_windows/1500_50/"
)

# Choose group: ALL, MAINLAND, ISLAND
GROUP = "ISLAND"

files = sorted(
    glob.glob(f"{RESULTS_DIR}/roh_gwas_windows_{GROUP}_*.tsv")
)

print(f"Found {len(files)} result files")

# ----------------------------
# Load all result files
# ----------------------------
df_IL = pd.concat(
    (pd.read_csv(f, sep="\t") for f in files),
    ignore_index=True
)

print(f"Total windows loaded (raw): {df_IL.shape[0]}")

# Load data
df_IL = df_IL.loc[
    df_IL["P_ROH_STATUS"].notna()
    & np.isfinite(df_IL["P_ROH_STATUS"])
].copy()

# Compute -log10(p) exactly as in the plot
df_IL["LOGP"] = -np.log10(df_IL["P_ROH_STATUS"])
print(f"Total windows loaded: {df_IL.shape[0]}")

neff_genome_IL = 1409  # your simpleM result

p_eff_IL = 0.05 / neff_genome_IL
logp_eff_IL = -np.log10(p_eff_IL)

print(f"Neff Bonferroni threshold:")
print(f"  p ≤ {p_eff_IL:.3e}")
print(f"  -log10(p) ≥ {logp_eff_IL:.4f}")

sig_df_IL = df_IL.loc[
    df_IL["LOGP"] >= logp_eff_IL
].copy()

print(f"Significant windows (Neff, LOGP-based): {sig_df_IL.shape[0]}")
print(sig_df_IL)
```


```python
df_IL["BETA_ROH_STATUS"].describe()
```


```python
is_sig = df_IL["LOGP"] >= logp_eff_IL
beta_min = -3
beta_max = 3

beta_ok = (
    df_IL["BETA_ROH_STATUS"].notna()
    & (df_IL["BETA_ROH_STATUS"] >= beta_min)
    & (df_IL["BETA_ROH_STATUS"] <= beta_max)
)
plot_mask = beta_ok
sig_plot_mask = beta_ok & is_sig

n_excluded = (~beta_ok).sum()
print(f"Effect sizes excluded from plot (|β| outside [{beta_min}, {beta_max}]): {n_excluded}")


plt.figure(figsize=(10, 4))

# Non-significant (within beta limits)
plt.scatter(
    df_IL.loc[plot_mask & ~is_sig].index,
    df_IL.loc[plot_mask & ~is_sig, "BETA_ROH_STATUS"],
    color="lightgrey",
    s=8,
    alpha=0.7,
    label="Not significant"
)

# Significant (within beta limits)
plt.scatter(
    df_IL.loc[sig_plot_mask].index,
    df_IL.loc[sig_plot_mask, "BETA_ROH_STATUS"],
    color="red",
    s=15,
    label="Neff significant"
)

plt.axhline(0, color="black", linewidth=0.8)

plt.ylim(beta_min, beta_max)

plt.xlabel("Window index")
plt.ylabel("Effect size (β_ROH_STATUS)")
plt.title("ROH-GWAS effect sizes (Island)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

```


```python
INPUT_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "roh_gwas_input_windows/1500_50/"
)

input_files = sorted(
    glob.glob(f"{INPUT_DIR}/gwas_input_windows_covariates_*.tsv")
)

df_input = pd.concat(
    (pd.read_csv(f, sep="\t") for f in input_files),
    ignore_index=True
)

print(f"Raw input rows loaded: {df_input.shape[0]}")

df_input_IL = df_input[df_input["PC1_IL"].notna()].copy()
df_input_IL["Population"] = df_input_IL["IID"].str.extract(r"([A-Z]+)")
df_input_IL["Population"] = df_input_IL["Population"].str.replace(r"^24", "", regex=True)

sig_windows = sig_df_IL["WINDOW_ID"].unique()
print(f"Significant windows: {len(sig_windows)}")

df_plot_IL = df_input_IL[df_input_IL["WINDOW_ID"].isin(sig_windows)].copy()

color_map = {
    'PJ':'#6BAED6', 'MP': '#2171B5', 'KP': '#4292C6', 'PM': '#08306B',
    'BJ':'#08519C', 'SC':'#9ECAE1', 'PK':'#C6DBEF', 'VS':'#238B45',
    'TM':'#00441B', 'RG':'#BAE4B3', 'SP':'#74C476', 'TR':'#006D2C',
    'PL':'#41AB5D'
}

windows = df_plot_IL["WINDOW_ID"].unique()
print(f"Plotting {len(windows)} significant windows")

for window in windows:

    df_w = df_plot_IL[df_plot_IL["WINDOW_ID"] == window]

    # Skip windows with no ROH variation
    if df_w["ROH_STATUS"].nunique() < 2:
        continue

    # Get effect size from GWAS results
    beta = sig_df_IL.loc[
        sig_df_IL["WINDOW_ID"] == window,
        "BETA_ROH_STATUS"
    ].iloc[0]

    plt.figure(figsize=(6, 4))

    # Boxplot
    sns.boxplot(
        x="ROH_STATUS",
        y="PHENO",     # or PHENO if you prefer raw scale
        data=df_w,
        showcaps=True,
        boxprops={"facecolor": "none"},
        width=0.5
    )

    # Population-coloured points
    sns.stripplot(
        x="ROH_STATUS",
        y="PHENO",
        data=df_w,
        hue="Population",
        dodge=True,
        jitter=True,
        size=6,
        alpha=0.8,
        palette=color_map
    )

    plt.xlabel("ROH status (0 = non-ROH, 1 = ROH)")
    plt.ylabel("Phenotype")
    plt.title(f"{window}\nROH effect β = {beta:.3f}")

    plt.legend(
        title="Population",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

```


```python
def plot_roh_windows_grid(
    df_plot,
    sig_df,
    color_map,
    effect_col="BETA_ROH_STATUS",
    fig_width_cm=20,   # <- wider overall
    fig_height_cm=5,
    n_cols=6,          # <- FEWER columns → wider panels
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    out_file=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    import math
    import os

    # =========================
    # Font (HOUSE STYLE)
    # =========================
    fm.fontManager.addfont(arial_path)
    arial_font = fm.FontProperties(fname=arial_path)
    arial_name = arial_font.get_name()

    plt.rcParams['font.family'] = arial_name
    plt.rcParams['font.sans-serif'] = [arial_name]

    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    # =========================
    # Log-transform phenotype
    # =========================
    df_plot = df_plot.copy()
    df_plot["PHENO_LOG"] = np.log(df_plot["PHENO"])

    # =========================
    # Windows
    # =========================
    windows = df_plot["WINDOW_ID"].unique()
    n_plots = len(windows)

    # enforce 2 rows
    n_rows = 2
    n_cols = math.ceil(n_plots / n_rows)

    # =========================
    # Figure
    # =========================
    cm = 1 / 2.54
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width_cm * cm, fig_height_cm * cm),
        dpi=300,
        squeeze=False
    )

    axes = axes.flatten()

    # consistent population order
    pop_order = sorted(color_map.keys())

    # =========================
    # Plot loop
    # =========================
    for i, window in enumerate(windows):

        ax = axes[i]
        df_w = df_plot[df_plot["WINDOW_ID"] == window]

        if df_w["ROH_STATUS"].nunique() < 2:
            ax.axis("off")
            continue

        beta = sig_df.loc[
            sig_df["WINDOW_ID"] == window,
            effect_col
        ].iloc[0]

        # Grid
        ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

        # Boxplot (log scale)
        sns.boxplot(
            x="ROH_STATUS",
            y="PHENO_LOG",
            data=df_w,
            showcaps=True,
            boxprops={"facecolor": "none", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            medianprops={"linewidth": 1},
            width=0.5,
            ax=ax
        )

        # Stripplot (NO jitter, grouped populations)
        sns.stripplot(
            x="ROH_STATUS",
            y="PHENO_LOG",
            data=df_w,
            hue="Population",
            hue_order=pop_order,
            palette=color_map,
            jitter=False,      # <- key change
            dodge=True,        # <- separates populations
            size=4,
            alpha=0.9,
            ax=ax,
            zorder=2
        )

        # remove per-panel legend
        if ax.get_legend():
            ax.get_legend().remove()

        # Labels
        ax.set_xlabel("ROHstatus (0/1)", fontsize=6,labelpad=0.5)
        ax.set_ylabel("Ln(Bite force)", fontsize=6,labelpad=0.5)
        # =========================
        # Format window label (3 lines)
        # =========================
        chrom, coords = window.split(":")
        start, end = coords.split("-")
        
        coords_fmt = f"{int(start)}–{int(end)}"
        
        title = f"{chrom}\n{coords_fmt}\nβ = {beta:.2f}"
        
        ax.set_title(title, fontsize=6, linespacing=1.2)

        ax.tick_params(axis='x', pad=1)
        ax.tick_params(axis='y', pad=1)

    # =========================
    # Remove empty panels
    # =========================
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")


    # # =========================
    # # GLOBAL LEGEND
    # # =========================
    # handles = [
    #     plt.Line2D([0], [0], marker='o', linestyle='', color=color_map[p], label=p, markersize=4)
    #     for p in pop_order
    # ]

    # fig.legend(
    #     handles=handles,
    #     labels=pop_order,
    #     title="Population",
    #     loc="lower right",
    #     bbox_to_anchor=(0.5, 0.98),
    #     ncol=len(pop_order),
    #     fontsize=6,
    #     frameon=False
    # )

    # =========================
    # Layout (HOUSE STYLE)
    # =========================
    plt.subplots_adjust(
        left=0.08,
        right=0.98,   # <- use full width
        bottom=0.12,
        top=0.88,
        wspace=0.4,
        hspace=0.6    # <- more vertical space for titles
    )

    # =========================
    # Save
    # =========================
    if out_file is not None:
        plt.savefig(out_file, dpi=300)
        print(f"Saved figure to: {os.path.abspath(out_file)}")

    plt.show()
```


```python
plot_roh_windows_grid(
    df_plot=df_plot_IL,
    sig_df=sig_df_IL,
    color_map=color_map1,
    out_file="ROH_windows_grid.svg"
)
```


```python
def plot_roh_windows_grid(
    df_plot,
    sig_df,
    color_map,
    effect_col="BETA_ROH_STATUS",
    fig_width_cm=20,   # <- wider overall
    fig_height_cm=5,
    n_cols=6,          # <- FEWER columns → wider panels
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    out_file=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    import math
    import os

    # =========================
    # Font (HOUSE STYLE)
    # =========================
    fm.fontManager.addfont(arial_path)
    arial_font = fm.FontProperties(fname=arial_path)
    arial_name = arial_font.get_name()

    plt.rcParams['font.family'] = arial_name
    plt.rcParams['font.sans-serif'] = [arial_name]

    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    # =========================
    # Log-transform phenotype
    # =========================
    df_plot = df_plot.copy()
    df_plot["PHENO_LOG"] = np.log(df_plot["PHENO"])

    # =========================
    # Windows
    # =========================
    windows = df_plot["WINDOW_ID"].unique()
    n_plots = len(windows)

    # enforce 2 rows
    n_rows = 1
    n_cols = n_plots

    # =========================
    # Figure
    # =========================
    cm = 1 / 2.54
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width_cm * cm, fig_height_cm * cm),
        dpi=300,
        squeeze=False
    )

    axes = axes.flatten()

    # consistent population order
    pop_order = sorted(color_map.keys())

    # =========================
    # Plot loop
    # =========================
    for i, window in enumerate(windows):

        ax = axes[i]
        df_w = df_plot[df_plot["WINDOW_ID"] == window]

        if df_w["ROH_STATUS"].nunique() < 2:
            ax.axis("off")
            continue

        beta = sig_df.loc[
            sig_df["WINDOW_ID"] == window,
            effect_col
        ].iloc[0]

        # Grid
        ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

        # Boxplot (log scale)
        sns.boxplot(
            x="ROH_STATUS",
            y="PHENO_LOG",
            data=df_w,
            showcaps=True,
            boxprops={"facecolor": "none", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            medianprops={"linewidth": 1},
            width=0.5,
            ax=ax
        )

        # Stripplot (NO jitter, grouped populations)
        sns.stripplot(
            x="ROH_STATUS",
            y="PHENO_LOG",
            data=df_w,
            hue="Population",
            hue_order=pop_order,
            palette=color_map,
            jitter=False,      # <- key change
            dodge=True,        # <- separates populations
            size=4,
            alpha=0.9,
            ax=ax,
            zorder=2
        )

        # remove per-panel legend
        if ax.get_legend():
            ax.get_legend().remove()

        # Labels
        ax.set_xlabel("ROHstatus (0/1)", fontsize=6,labelpad=0.5)
        ax.set_ylabel("Ln(Bite force)", fontsize=6,labelpad=0.5)
        # =========================
        # Format window label (3 lines)
        # =========================
        chrom, coords = window.split(":")
        start, end = coords.split("-")
        
        coords_fmt = f"{int(start)}–{int(end)}"
        
        title = f"{chrom}\n{coords_fmt}\nβ = {beta:.2f}"
        
        ax.set_title(title, fontsize=6, linespacing=1.2)

        ax.tick_params(axis='x', pad=1)
        ax.tick_params(axis='y', pad=1)

    # =========================
    # Remove empty panels
    # =========================
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")


    # # =========================
    # # GLOBAL LEGEND
    # # =========================
    # handles = [
    #     plt.Line2D([0], [0], marker='o', linestyle='', color=color_map[p], label=p, markersize=4)
    #     for p in pop_order
    # ]

    # fig.legend(
    #     handles=handles,
    #     labels=pop_order,
    #     title="Population",
    #     loc="lower right",
    #     bbox_to_anchor=(0.5, 0.98),
    #     ncol=len(pop_order),
    #     fontsize=6,
    #     frameon=False
    # )

    # =========================
    # Layout (HOUSE STYLE)
    # =========================
    plt.subplots_adjust(
        left=0.08,
        right=0.74,   # <- use full width
        bottom=0.12,
        top=0.74,
        wspace=0.4,
        hspace=0.6    # <- more vertical space for titles
    )

    # =========================
    # Save
    # =========================
    if out_file is not None:
        plt.savefig(out_file, dpi=300)
        print(f"Saved figure to: {os.path.abspath(out_file)}")

    plt.show()
```


```python
plot_roh_windows_grid(
    df_plot=df_plot_ML,
    sig_df=sig_df_ML,
    color_map=color_map2,
    out_file="ROH_windows_grid_ML.svg"
)
```


```python
color_map2 = {'VS': '#0A592A',
    'TM': '#3FAD5B'}
```


```python

```


```python
sig_df_IL[["WINDOW_ID", "CHR", "BETA_ROH_STATUS", "N_ROH", "N_NONROH", "P_ROH_STATUS", "LOGP"]] \
    .to_csv("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_50/significant_windows/roh_neff_significant_windows_IL.tsv", sep="\t", index=False)

```

#### 11.1.3. Plots


```python
def plot_manhattan_plot(
    file_glob,
    title,
    neff_genome,
    sig_file=None,                 # <--- NEW
    sig_effect_col="BETA_ROH_STATUS",         # effect size column
    chrom_colors=("#4C72B0", "#DD8452"),
    fig_width_cm=10,
    fig_height_cm=5,
    point_size=10,
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    out_file=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import glob

    # ==================================================
    # Font: Arial
    # ==================================================
    arial_font = fm.FontProperties(fname=arial_path)
    plt.rcParams["font.family"] = "Dejavu Sans"

    # ==================================================
    # Figure size
    # ==================================================
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(fig_width_cm * cm, fig_height_cm * cm))

    # ==================================================
    # Load GWAS windows
    # ==================================================
    files = sorted(glob.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {file_glob}")

    df = pd.concat(
        (pd.read_csv(f, sep="\t") for f in files),
        ignore_index=True
    )

    df["POS"] = df["WINDOW_ID"].apply(window_midpoint)
    df["LOGP"] = -np.log10(df["P_ROH_STATUS"])

    # ==================================================
    # Chromosome order & cumulative position
    # ==================================================
    chrom_order = sorted(df["CHR"].unique(), key=chrom_numeric)

    chrom_offsets, offset = {}, 0
    for chrom in chrom_order:
        chrom_offsets[chrom] = offset
        offset += df.loc[df["CHR"] == chrom, "POS"].max()

    df["POS_CUM"] = df.apply(
        lambda r: r["POS"] + chrom_offsets[r["CHR"]],
        axis=1
    )

    chrom_centers = [
        (df[df["CHR"] == chrom]["POS_CUM"].min()
         + df[df["CHR"] == chrom]["POS_CUM"].max()) / 2
        for chrom in chrom_order
    ]

    # ==================================================
    # Threshold
    # ==================================================
    eff_bonf_logp = -np.log10(0.05 / neff_genome)

    # ==================================================
    # Main scatter
    # ==================================================
    for i, chrom in enumerate(chrom_order):
        sub = df[df["CHR"] == chrom]
        ax.scatter(
            sub["POS_CUM"],
            sub["LOGP"],
            s=point_size,
            color=chrom_colors[i % len(chrom_colors)],
            edgecolor="none",
            zorder=2
        )

    ax.axhline(
        eff_bonf_logp,
        color="black",
        linestyle="--",
        linewidth=3,
        zorder=1
    )

    # ==================================================
    # Overlay significant hits (optional)
    # ==================================================
    if sig_file is not None:

        # Load sig_df
        if isinstance(sig_file, str):
            sig_df = pd.read_csv(sig_file, sep=None, engine="python")
        else:
            sig_df = sig_file.copy()

        # Merge to get plotting coordinates
        sig_df = sig_df.merge(
            df[["WINDOW_ID", "POS_CUM"]],
            on="WINDOW_ID",
            how="left"
        )


        sig_pos = sig_df[sig_df[sig_effect_col] > 0]
        sig_neg = sig_df[sig_df[sig_effect_col] < 0]

        ax.scatter(
            sig_pos["POS_CUM"],
            sig_pos["LOGP"],
            s=point_size * 1.6,
            color="#7B1FA2",
            edgecolor="#7B1FA2",
            linewidth=0.3,
            zorder=4,
            label="Positive effect"
        )

        ax.scatter(
            sig_neg["POS_CUM"],
            sig_neg["LOGP"],
            s=point_size * 1.6,
            color="#B22222",
            edgecolor="#B22222",
            linewidth=0.3,
            zorder=4,
            label="Negative effect"
        )

    # ==================================================
    # Axis formatting
    # ==================================================
    chrom_labels = [str(i + 1) for i in range(len(chrom_centers))]

    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(chrom_labels, fontsize=20)

    ax.tick_params(axis="y", labelsize=30)

    ax.set_xlabel("Chromosome", fontsize=30, fontproperties=arial_font)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=40, fontproperties=arial_font)
    ax.set_title(title)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=2, color="grey", alpha=0.7)
    # ax.minorticks_on()
    # ax.grid(
    #     which="minor",
    #     axis="y",
    #     linestyle="--",
    #     linewidth=2,
    #     color="grey",
    #     alpha=0.5
    # )

    # Start dots at y=0
    ax.set_ylim(bottom=0)
    ax.margins(y=0, x=0.01)

    # Black box
    # for spine in ax.spines.values():
    #     spine.set_color("black")
    #     spine.set_linewidth(1)

    # Legend
    if sig_file is not None:
        ax.legend().remove()
    else:
        ax.legend().remove()

    plt.tight_layout()
    # ==================================================
    # Save figure (optional)
    # ==================================================
    if out_file is not None:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {os.path.abspath(out_file)}")

    plt.show()

```


```python
plot_manhattan_plot(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_MAINLAND_*.tsv",
    title="",
    sig_file=sig_df_ML,
    neff_genome=947,
    chrom_colors=("#009900", "#74C476"),
    fig_width_cm=38,
    fig_height_cm=20,
    point_size=100,
    out_file="ROH_GWAS_ML_manhattan_poster.png"
)
```


```python
plot_manhattan_plot(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_ISLAND_*.tsv",
    title="",
    sig_file=sig_df_IL,
    neff_genome=1409,
    chrom_colors=("#0060CA", "#6BAED6"),
    fig_width_cm=38,
    fig_height_cm=20,
    point_size=100,
    out_file="ROH_GWAS_IL_manhattan_poster.png"
)

```


```python
plot_manhattan_plot(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_ISLAND_*.tsv",
    title="",
    sig_file=sig_df_IL,
    neff_genome=1409,
    chrom_colors=("#0060CA", "#6BAED6"),
    fig_width_cm=20,
    fig_height_cm=10,
    out_file="ROH_GWAS_IL_manhattan.png"
)

```


```python
def plot_effect_size_bellcurve_filled_with_sig_points(
    file_glob,
    sig_df,
    effect_col="BETA_ROH_STATUS",
    fig_width_cm=6.75,
    fig_height_cm=6,
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    neg_color="#479BC9",   # <- MATCH
    pos_color="#208438",   # <- MATCH
    curve_alpha=0.5,
    point_size=6,
    out_file=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import glob
    import os
    from scipy.stats import gaussian_kde

    # =========================
    # Font (IDENTICAL SYSTEM)
    # =========================
    fm.fontManager.addfont(arial_path)
    arial_font = fm.FontProperties(fname=arial_path)
    arial_name = arial_font.get_name()

    plt.rcParams['font.family'] = arial_name
    plt.rcParams['font.sans-serif'] = [arial_name]

    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    # =========================
    # Figure (IDENTICAL SIZE)
    # =========================
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(fig_width_cm * cm, fig_height_cm * cm),
        dpi=300
    )

    # =========================
    # Load data
    # =========================
    files = sorted(glob.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {file_glob}")

    df_all = pd.concat(
        (pd.read_csv(f, sep="\t") for f in files),
        ignore_index=True
    )

    effects_all = df_all[effect_col].dropna().values

    # =========================
    # KDE
    # =========================
    kde = gaussian_kde(effects_all)
    x = np.linspace(
        effects_all.min() * 1.1,
        effects_all.max() * 1.1,
        1000
    )
    y = kde(x)

    x_neg, y_neg = x[x <= 0], y[x <= 0]
    x_pos, y_pos = x[x >= 0], y[x >= 0]

    # =========================
    # Significant points
    # =========================
    sig_effects = sig_df[effect_col].dropna().values
    pos = sig_effects > 0
    neg = sig_effects < 0

    y_jitter = np.random.uniform(
        low=0,
        high=y.max() * 0.04,
        size=len(sig_effects)
    )

    # =========================
    # Grid (MATCHED)
    # =========================
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

    # =========================
    # KDE fill (soft)
    # =========================
    ax.fill_between(x_neg, y_neg, color=neg_color, alpha=curve_alpha)
    ax.fill_between(x_pos, y_pos, color=pos_color, alpha=curve_alpha)

    # outline (subtle)
    ax.plot(x, y, color="black", linewidth=0.5)

    # zero line
    ax.axvline(0, linestyle="--", linewidth=0.5, color="black")

    # =========================
    # Points (MATCH STYLE)
    # =========================
    ax.scatter(
        sig_effects[pos],
        y_jitter[pos],
        color=pos_color,
        s=point_size,
        alpha=1,
        zorder=2
    )

    ax.scatter(
        sig_effects[neg],
        y_jitter[neg],
        color=neg_color,
        s=point_size,
        alpha=1,
        zorder=2
    )

    # =========================
    # Axes styling (MATCHED)
    # =========================
    ax.set_xlabel("")
    ax.set_ylabel("")

    # remove y labels
    ax.tick_params(axis='y', labelleft=False)

    ax.tick_params(axis='x', pad=4, labelbottom=False)
    ax.tick_params(axis='y', pad=4)

    ax.margins(x=0.02)

    # =========================
    # Layout (CRUCIAL MATCH)
    # =========================
    ax.set_position([0.10, 0.18, 0.85, 0.74])

    # =========================
    # Save (FIXED)
    # =========================
    fig.canvas.draw()

    if out_file is not None:
        plt.savefig(out_file, dpi=300)  # NO tight bbox
        print(f"Saved figure to: {os.path.abspath(out_file)}")

    plt.show()
```


```python
plot_effect_size_bellcurve_filled_with_sig_points(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
              "gwas_roh/roh_gwas_results_windows/1500_50/"
              "roh_gwas_windows_ISLAND_*.tsv",
    sig_df = sig_df_IL,
    effect_col="BETA_ROH_STATUS",
    neg_color="#B22222",   
    pos_color="#7B1FA2",
    out_file="ROH_GWAS_IL_effect.png"
    
)

```


```python
plot_effect_size_bellcurve_filled_with_sig_points(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
              "gwas_roh/roh_gwas_results_windows/1500_50/"
              "roh_gwas_windows_MAINLAND_*.tsv",
    sig_df = sig_df_ML,
    effect_col="BETA_ROH_STATUS",
    neg_color="#B22222",   
    pos_color="#7B1FA2",
    out_file="ROH_GWAS_ML_effect.png"
)

```


```python

```

## 13. Plot inbreeding depression


```python
beta_df_std = pd.DataFrame([
    # trait_group, trait, population, sex, beta_mean, beta_low, beta_high

    # ----------------------------
    # Physiological performance
    # ----------------------------
    ["Physiological performance", "Bite force", "Island", None,
     -2.0825, -3.3115, -0.9022],

    ["Physiological performance", "Bite force", "Mainland", None,
     -2.0825 + 4.8643,
     -3.3115 - 0.5051,
     -0.9022 + 10.0810],
    
    ["Physiological performance", "Sprint speed", "Island", None,
     -0.666459, -2.576064, 1.226646],

    ["Physiological performance", "Sprint speed", "Mainland", None,
      1.570748, -8.789315, 11.841722],
    ["Sperm quality", "Sperm count", "Island", None,
      0.93628,  -1.27815,   3.41029],

    ["Sperm quality", "Sperm count", "Mainland", None,
     -2.80918, -15.93617,  10.01102],
    ["Sperm quality", "Sperm length", "Island", None,
      -2.9402315,  -9.7979168,   2.9294237],

    ["Sperm quality", "Sperm length", "Mainland", None,
       6.9772658, -11.6636482,  25.4322047],
    ["Sperm quality", "Sperm velocity", "Island", None,
      -3.19861,  -7.48717,   0.10589],

    ["Sperm quality", "Sperm velocity", "Mainland", None,
       6.06658, -10.15141,  21.56426],

    ["Immune function", "PHA", "Island", None,
      0.55276,  -1.64983,   3.23183],

    ["Immune function", "PHA", "Mainland", None,
      3.87819,  -9.83934,  17.96221],
    ["Cognition", "Spatial learning", "Island", None,
      -0.81946,  -2.13903,   0.60291],

    ["Cognition", "Spatial learning", "Mainland", None,
       0.92636,  -6.08416,   7.46533],

],
columns=[
    "trait_group", "trait", "population", "sex",
    "beta_mean", "beta_low", "beta_high"
])

beta_df_std
```


```python
def plot_beta_combined(
    df,
    figsize_cm=(9.53, 9.15),
    xlim=(-26, 26)
):

    import numpy as np
    import matplotlib.pyplot as plt

    order = df["trait"].unique()[::-1]
    y_positions = {trait: i for i, trait in enumerate(order)}
    offset = {"Island": -0.15, "Mainland": +0.15}

    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(figsize_cm[0]*cm, figsize_cm[1]*cm),
        dpi=300
    )

    # =========================
    # Plot (SOLID POINTS)
    # =========================
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=2)
    for _, row in df.iterrows():

        y = y_positions[row["trait"]] + offset[row["population"]]

        ax.errorbar(
            row["beta_mean"],
            y,
            xerr=[[row["beta_mean"] - row["beta_low"]],
                  [row["beta_high"] - row["beta_mean"]]],
            fmt="o",
            color=palette[row["population"]],
            capsize=3,
            markersize=4,
            linewidth=1.2,
            alpha=1.0,
            zorder=2
        )

    # =========================
    # Grid
    # =========================
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.grid(axis="y", visible=False)

    # remove grid line at x = 0
    ax.axvline(0, color="white", linewidth=1.2, zorder=1)

    # =========================
    # Reference line (clean)
    # =========================


    # =========================
    # Y axis (structure only)
    # =========================
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])

    # =========================
    # X axis
    # =========================
    ax.set_xlim(xlim)
    ax.set_xticklabels([])

    # =========================
    # Remove axis labels
    # =========================
    ax.set_xlabel("")
    ax.set_ylabel("")

    # =========================
    # Ticks
    # =========================
    ax.tick_params(
        axis='x',
        bottom=True,
        length=3,
        width=0.8,
        direction='out'
    )

    ax.tick_params(
        axis='y',
        left=True,
        length=3,
        width=0.8,
        direction='out'
    )

    # =========================
    # Spines
    # =========================
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # =========================
    # Layout
    # =========================
    ax.set_position([0.25, 0.12, 0.7, 0.82])

    # =========================
    # Legend (tight dots only)
    # =========================
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=palette[k],
                   markeredgecolor=palette[k],
                   markersize=5,
                   label="")
        for k in palette
    ]

    ax.legend(
        handles=handles,
        labels=["        ", "        "],
        frameon=False,
        loc="lower right",
        handlelength=0,
        handletextpad=0,
        borderpad=0,
        labelspacing=0.05,
        handleheight=0.1
    )

    return fig, ax
```


```python
fig, ax = plot_beta_combined(
    beta_df_std,
    xlim=(-28, 28)                       # optional
)

plt.savefig("beta_combined1.svg", dpi=300)
plt.show()
```


```python

```


```python
import pandas as pd

# ---- Load posterior samples ----
post = pd.read_csv(
    "mcmcglmm_posterior_fixed_effects.tsv",
    sep="\t"
)

# ---- Load predictions ----
pred = pd.read_csv(
    "mcmcglmm_BF_Froh_predictions.tsv",
    sep="\t"
)

pred.head()

import pandas as pd

df = pd.read_csv(
    "mcmcglmm_raw_data_BF_Froh.tsv",
    sep="\t"
)

df.head()


```


```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# =========================
# Load and register Arial (MATCHED)
# =========================
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"

fm.fontManager.addfont(arial_path)
arial_font = fm.FontProperties(fname=arial_path)
arial_name = arial_font.get_name()

plt.rcParams['font.family'] = arial_name
plt.rcParams['font.sans-serif'] = [arial_name]

plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

# =========================
# Figure size (MATCH SYSTEM)
# =========================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(9.53*cm, 4.61*cm), dpi=300)

# =========================
# Colors
# =========================
color = "#479BC9"

# =========================
# Raw data
# =========================
d_raw = df[df["type"] == "island"]

ax.scatter(
    d_raw["Froh"],
    d_raw["log_BF_max"],
    s=10,
    alpha=0.5,
    color=color,
    edgecolor="none",
    zorder=1
)

# =========================
# Predictions
# =========================
pred_sub = pred[(pred["Froh"] >= 0.505) & (pred["Froh"] <= 0.70)]
d = pred_sub[pred_sub["type"] == "island"]

ax.fill_between(
    d["Froh"],
    np.log(d["BF_low"]),
    np.log(d["BF_high"]),
    alpha=0.3,
    color=color,
    zorder=2
)

ax.plot(
    d["Froh"],
    np.log(d["BF_mean"]),
    color=color,
    linewidth=2.0,
    zorder=3
)

# =========================
# Grid
# =========================
ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
ax.grid(axis="y", visible=False)

# =========================
# Axes
# =========================
ax.set_xlim(0.50, 0.65)

# REMOVE LABELS
ax.set_xlabel("")
ax.set_ylabel("")

# =========================
# Tick styling
# =========================
ax.tick_params(
    axis='x',
    bottom=True,
    length=3,
    width=0.8,
    direction='out',
    pad=2
)

ax.tick_params(
    axis='y',
    left=True,
    length=3,
    width=0.8,
    direction='out',
    pad=2
)

# =========================
# Spines
# =========================
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.8)

# =========================
# Layout
# =========================
ax.set_position([0.15, 0.18, 0.80, 0.74])
ax.set_xticklabels([])
ax.set_yticklabels([])

# =========================
# Save
# =========================
fig.canvas.draw()

plt.savefig(
    "island_biteforce_panel.svg",
    format="svg",
    dpi=300
)

plt.show()
```


```python

```

## 15. ROH correlations among individuals


```python
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re
```


```python
files = sorted(glob.glob("../results/population_analyses/rPodSic1.hap1.1//genetic_diversity/bcftools_roh/ind_roh_windows/all_samples_windows_rohs/roh_windows_*.tsv"))

dfs = []

for f in files:
    df_rohs_cor = pd.read_csv(f, sep="\t")
    
    chrom = f.split("roh_windows_")[1].split("_")[0]
    df_rohs_cor["chrom"] = chrom
    
    dfs.append(df_rohs_cor)

df_rohs_cor = pd.concat(dfs, ignore_index=True)

print(df_rohs_cor.shape)
df_rohs_cor.head()
```


```python
# exclude WINDOW_ID and chrom
sample_cols = df_rohs_cor.columns[1:-1]

len(sample_cols)  # should be 263

def get_population(sample):
    pop = ''.join(re.findall(r'[A-Z]+', sample))
    pop = re.sub(r'^24', '', pop)
    if pop in ["CV", "FV"]:
        pop = "RG"
    return pop

def assign_region(pop):
    if pop in ["PL", "TR", "SP", "TM", "RG", "VS"]:
        return "Mainland"
    elif pop in ["MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]:
        return "Island"
    else:
        return "Unknown"

sample_info = pd.DataFrame({"sample": sample_cols})
sample_info["Population"] = sample_info["sample"].apply(get_population)
sample_info["Region"] = sample_info["Population"].apply(assign_region)

sample_info.head()
```


```python
island_inds = sample_info.loc[sample_info["Region"] == "Island", "sample"]
# Sum across island individuals
roh_counts = df_rohs_cor[island_inds].sum(axis=1)

# Number of island individuals
n_island = len(island_inds)

# Windows where ALL individuals are in ROH
always_roh = roh_counts == n_island
fraction = always_roh.mean()
percentage = fraction * 100

print(f"{percentage:.2f}% of the genome is in ROH in all island individuals")
```


```python
always_roh.sum()
```


```python
mainland_samples = sample_info[sample_info["Region"] == "Mainland"]["sample"].tolist()
island_samples   = sample_info[sample_info["Region"] == "Island"]["sample"].tolist()

print(len(mainland_samples), len(island_samples))
```


```python
# convert to numeric just in case
df_rohs_cor[sample_cols] = df_rohs_cor[sample_cols].apply(pd.to_numeric, errors='coerce')

# compute frequencies
df_rohs_cor["roh_all"] = (df_rohs_cor[sample_cols] > 0).mean(axis=1)
df_rohs_cor["roh_mainland"] = (df_rohs_cor[mainland_samples] > 0).mean(axis=1)
df_rohs_cor["roh_island"] = (df_rohs_cor[island_samples] > 0).mean(axis=1)
```


```python
coords = df_rohs_cor["WINDOW_ID"].str.extract(r":(\d+)-(\d+)")

df_rohs_cor["start"] = coords[0].astype(int)
df_rohs_cor["end"] = coords[1].astype(int)
df_rohs_cor["mid"] = (df_rohs_cor["start"] + df_rohs_cor["end"]) / 2 / 1e6
```


```python
import re

# Extract start and end positions
df_rohs_cor[["start", "end"]] = df_rohs_cor["WINDOW_ID"].str.extract(r":(\d+)-(\d+)").astype(int)
```


```python
df_rohs_cor
```


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================
# USER-DEFINED FIGURE SIZE
# =========================
width_cm = 15
height_cm = 8

# =========================
# Load Arial
# =========================
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
fm.fontManager.addfont(arial_path)
arial_font = fm.FontProperties(fname=arial_path)
arial_name = arial_font.get_name()

plt.rcParams['font.family'] = arial_name
plt.rcParams['font.sans-serif'] = [arial_name]

plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

sns.set_style("white")

# =========================
# Prepare coordinates
# =========================
df_rohs_cor[["start", "end"]] = df_rohs_cor["WINDOW_ID"].str.extract(r":(\d+)-(\d+)").astype(int)

# =========================
# Figure layout
# =========================
cm = 1 / 2.54
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(width_cm*cm, height_cm*cm),
    dpi=300,
    sharex=True,
    sharey=True
)

regions = ["roh_mainland", "roh_island"]

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=100)

# =========================
# Plot panels
# =========================
for i, region in enumerate(regions):

    ax = axes[i]
    chroms = sorted(df_rohs_cor["chrom"].unique())

    for j, chrom in enumerate(chroms):

        sub = df_rohs_cor[df_rohs_cor["chrom"] == chrom].sort_values("start")

        x = np.concatenate([
            sub["start"].values / 1e6,
            [sub["end"].values[-1] / 1e6]
        ])

        y = np.array([j - 0.4, j + 0.4])
        z = (sub[region].values * 100)[np.newaxis, :]

        ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='auto')

    # ---- Axes styling ----
    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(range(1, len(chroms)+1))

    ax.tick_params(axis='x', bottom=True, length=3, width=0.8, direction='out', pad=2)
    ax.tick_params(axis='y', left=True, length=3, width=0.8, direction='out', pad=2)

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)

    ax.set_title(region.replace("roh_", "").capitalize(), fontsize=7)

    # =========================
    # Density inset (NEW)
    # =========================
    axins = inset_axes(ax, width="35%", height="30%", loc="upper right", borderpad=1)

    sns.kdeplot(
        df_rohs_cor[region] * 100,
        fill=True,
        ax=axins,
        color="black",
        linewidth=1
    )

    axins.set_xlabel("% ROH", fontsize=6)
    axins.set_ylabel("Density", fontsize=6)
    axins.tick_params(axis='both', labelsize=5, length=2)

    for spine in ["top", "right"]:
        axins.spines[spine].set_visible(False)

# =========================
# Labels
# =========================
for ax in axes:
    ax.set_xlabel("Position (Mb)")

axes[0].set_ylabel("Chromosome")

# =========================
# Colorbar (moved right)
# =========================
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    orientation='vertical',
    fraction=0.04,
    pad=0.08   # <-- increased spacing (KEY CHANGE)
)

cbar.set_label("% individuals in ROH", fontsize=7)
cbar.ax.tick_params(labelsize=6)

# =========================
# Layout tuning
# =========================
plt.subplots_adjust(
    left=0.12,
    right=0.82,   # <-- more space for colorbar
    top=0.88,
    bottom=0.20,
    wspace=0.15
)

# =========================
# Save
# =========================
plt.savefig("Figure_ROH_landscape_2panel.png", format="png", dpi=300)

plt.show()
```


```python

```


```python

```


```python

```
