# Analyses and plots genetic load manuscript


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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Create basic Manhattan plot
plt.figure(figsize=(14, 6))

# Use simple numeric index for x-axis
x_positions = range(len(df_SPC))
p_values = -np.log10(df_SPC['p_lrt'])

# Color by chromosome
unique_chroms = df_SPC['chr'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chroms)))

for i, chrom in enumerate(unique_chroms):
    chrom_mask = df_SPC['chr'] == chrom
    plt.scatter(np.array(x_positions)[chrom_mask], 
                p_values[chrom_mask], 
                color=colors[i], 
                s=10, 
                alpha=0.7,
                label=str(chrom))

# Add significance lines
plt.axhline(y=-np.log10(5e-8), color='grey', linestyle='--', linewidth=1, label='Genome-wide (5e-8)')
plt.axhline(y=bonferroni_line, color='red', linestyle='--', linewidth=1,
            label=f'Bonferroni (0.05/{n_tests:,} = {alpha:.2e})')

plt.xlabel('SNP Position')
plt.ylabel('-log10(P-value) - Likelihood Ratio Test')
plt.title('GEMMA LRT P-values SPERMCOUNT')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```


```python

```


```python
##### SimpleM all samples
```


```python
import pandas as pd
import numpy as np
import glob, os


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

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(eigvals)[::-1]

    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    neff = np.searchsorted(cumvar, variance_threshold) + 1

    return neff


def compute_neff_from_window_folder_all(
    input_dir,
    variance_threshold=0.995,
    file_pattern="gwas_input_windows_covariates_*.tsv"
):
    """
    Compute genome-wide N_eff using ALL individuals together.
    """

    files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))

    if not files:
        raise FileNotFoundError("No input files found in directory")

    total_neff = 0
    summary = []

    print(f"\nComputing N_eff using ALL individuals")
    print(f"Scanning {len(files)} chromosome files\n")

    for f in files:
        df = pd.read_csv(f, sep="\t")

        if df.empty:
            continue

        chrom = df["CHR"].iloc[0]

        # Individual × window matrix (ALL IIDs)
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
            "N_individuals": X.shape[0],
            "N_windows": X.shape[1],
            "N_eff": neff_chr
        })

        print(
            f"{chrom}: individuals={X.shape[0]} "
            f"windows={X.shape[1]}  N_eff={neff_chr}"
        )

    print(f"\nTOTAL genome-wide N_eff = {total_neff}\n")

    return total_neff, pd.DataFrame(summary)

```


```python
INPUT_DIR = "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/roh_gwas_input_windows/1500_50"

neff_all, neff_all_chr = compute_neff_from_window_folder_all(
    INPUT_DIR,
)
```


```python
##### Exploratory plots
```


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_ALL_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (ALL) – 1500 SNPs, 50% threshold",
    neff_genome=2097
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_50/roh_gwas_windows_ALL_*.tsv"),
    "ROH-GWAS QQ plot (ALL) - 1500 snps, 50% threshold"
)

```


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "gwas_roh/roh_gwas_results_windows/1500_50/"
)

# Choose group: ALL, MAINLAND, ISLAND
GROUP = "ALL"

files = sorted(
    glob.glob(f"{RESULTS_DIR}/roh_gwas_windows_{GROUP}_*.tsv")
)

print(f"Found {len(files)} result files")

# ----------------------------
# Load all result files
# ----------------------------
df_ALL = pd.concat(
    (pd.read_csv(f, sep="\t") for f in files),
    ignore_index=True
)

print(f"Total windows loaded (raw): {df_ALL.shape[0]}")

# Load data
df_ALL = df_ALL.loc[
    df_ALL["P_ROH_STATUS"].notna()
    & np.isfinite(df_ALL["P_ROH_STATUS"])
].copy()

# Compute -log10(p) exactly as in the plot
df_ALL["LOGP"] = -np.log10(df_ALL["P_ROH_STATUS"])
print(f"Total windows loaded: {df.shape[0]}")

neff_genome_ALL = 2097  # your simpleM result

p_eff_ALL = 0.05 / neff_genome_ALL
logp_eff_ALL = -np.log10(p_eff_ALL)

print(f"Neff Bonferroni threshold:")
print(f"  p ≤ {p_eff_ALL:.3e}")
print(f"  -log10(p) ≥ {logp_eff_ALL:.4f}")

sig_df_ALL = df_ALL.loc[
    df_ALL["LOGP"] >= logp_eff_ALL
].copy()

print(f"Significant windows (Neff, LOGP-based): {sig_df_ALL.shape[0]}")
print(sig_df_ALL)
```


```python
is_sig = df_ALL["LOGP"] >= logp_eff_ALL
beta_min = -10
beta_max = 10

beta_ok = (
    df_ALL["BETA_ROH_STATUS"].notna()
    & (df_ALL["BETA_ROH_STATUS"] >= beta_min)
    & (df_ALL["BETA_ROH_STATUS"] <= beta_max)
)
plot_mask = beta_ok
sig_plot_mask = beta_ok & is_sig

n_excluded = (~beta_ok).sum()
print(f"Effect sizes excluded from plot (|β| outside [{beta_min}, {beta_max}]): {n_excluded}")


plt.figure(figsize=(10, 4))

# Non-significant (within beta limits)
plt.scatter(
    df_ALL.loc[plot_mask & ~is_sig].index,
    df_ALL.loc[plot_mask & ~is_sig, "BETA_ROH_STATUS"],
    color="lightgrey",
    s=8,
    alpha=0.7,
    label="Not significant"
)

# Significant (within beta limits)
plt.scatter(
    df_ALL.loc[sig_plot_mask].index,
    df_ALL.loc[sig_plot_mask, "BETA_ROH_STATUS"],
    color="red",
    s=15,
    label="Neff significant"
)

plt.axhline(0, color="black", linewidth=0.8)

plt.ylim(beta_min, beta_max)

plt.xlabel("Window index")
plt.ylabel("Effect size (β_ROH_STATUS)")
plt.title("ROH-GWAS effect sizes (ALL)")
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

df_input_ALL = df_input[df_input["PC1"].notna()].copy()
df_input_ALL["Population"] = df_input_ALL["IID"].str.extract(r"([A-Z]+)")
df_input_ALL["Population"] = df_input_ALL["Population"].str.replace(r"^24", "", regex=True)

sig_windows = sig_df_ALL["WINDOW_ID"].unique()
print(f"Significant windows: {len(sig_windows)}")

df_plot_ALL = df_input_ALL[df_input_ALL["WINDOW_ID"].isin(sig_windows)].copy()

color_map = {
    'PJ':'#6BAED6', 'MP': '#2171B5', 'KP': '#4292C6', 'PM': '#08306B',
    'BJ':'#08519C', 'SC':'#9ECAE1', 'PK':'#C6DBEF', 'VS':'#238B45',
    'TM':'#00441B', 'RG':'#BAE4B3', 'SP':'#74C476', 'TR':'#006D2C',
    'PL':'#41AB5D'
}

windows = df_plot_ALL["WINDOW_ID"].unique()
print(f"Plotting {len(windows)} significant windows")

for window in windows:

    df_w = df_plot_ALL[df_plot_ALL["WINDOW_ID"] == window]

    # Skip windows with no ROH variation
    if df_w["ROH_STATUS"].nunique() < 2:
        continue

    # Get effect size from GWAS results
    beta = sig_df_ALL.loc[
        sig_df_ALL["WINDOW_ID"] == window,
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

```

## 14. Genetic load

### 14.1. SNPeff output


```python
df_load = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/genetic_load_counts.tsv", sep="\t")
df_load.head()
```


```python
df_load.groupby(["population_type","category"])[["Hom","Het","Tot"]].mean()
```


```python
df_ratio = df_load.pivot_table(
    index=["sample", "population", "population_type"],
    columns="category",
    values=["Tot", "Hom", "Het"]
).reset_index()
```


```python
df_ratio.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in df_ratio.columns
]
```


```python
df_ratio["DEL_SYN"] = df_ratio["Tot_DEL"] / df_ratio["Tot_SYN"]

df_ratio["LOF_SYN"] = df_ratio["Tot_LOF"] / df_ratio["Tot_SYN"]

df_ratio["HomDEL_HomSYN"] = df_ratio["Hom_DEL"] / df_ratio["Hom_SYN"]

df_ratio["HetDEL_HetSYN"] = df_ratio["Het_DEL"] / df_ratio["Het_SYN"]

df_ratio["HomLOF_HomSYN"] = df_ratio["Hom_LOF"] / df_ratio["Hom_SYN"]

df_ratio["HetLOF_HetSYN"] = df_ratio["Het_LOF"] / df_ratio["Het_SYN"]
```


```python
df_ratio.head()
```


```python
df_ratio_plot = df_ratio.melt(
    id_vars=["sample", "population", "population_type"],
    value_vars=["DEL_SYN", "LOF_SYN", "HomDEL_HomSYN", "HetDEL_HetSYN", "HomLOF_HomSYN","HetLOF_HetSYN"],
    var_name="metric",
    value_name="ratio"
)
```


```python
df_ratio_plot = df_ratio.melt(
    id_vars=["sample", "population", "population_type"],
    value_vars=["HomDEL_HomSYN", "HetDEL_HetSYN", "HomLOF_HomSYN","HetLOF_HetSYN"],
    var_name="metric",
    value_name="ratio"
)

df_ratio_plot["zygosity"] = df_ratio_plot["metric"].str.extract(r'^(Hom|Het)')[0].str.lower()

df_ratio_plot["metric_type"] = df_ratio_plot["metric"].str.extract(r'(DEL|LOF)')[0].map({
    "DEL": "deleterious",
    "LOF": "lof"
})
df_ratio_plot.head()
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

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

# =========================
# Figure (square)
# =========================
cm = 1 / 2.54
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(9.53*cm, 7.51*cm),   # <-- square figure
    dpi=300,
    sharex=True,
    sharey=False
)

# =========================
# Colors
# =========================
palette = {"mainland": "#208438", "island": "#479BC9"}

order = ["mainland", "island"]
metrics = ["deleterious", "lof"]
zyg = ["hom", "het"]

# =========================
# Loop over panels
# =========================
for i, metric in enumerate(metrics):
    for j, z in enumerate(zyg):

        ax = axes[i, j]

        sub = df_ratio_plot[
            (df_ratio_plot["metric_type"] == metric) &
            (df_ratio_plot["zygosity"] == z)
        ]

        # Grid
        ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

        # Stripplot
        sns.stripplot(
            data=sub,
            x="population_type",
            y="ratio",
            hue="population_type",
            palette=palette,
            order=order,
            jitter=True,
            size=5,
            alpha=0.5,
            dodge=False,
            zorder=1,
            ax=ax
        )

        # Remove legend
        if ax.get_legend():
            ax.get_legend().remove()

        # =========================
        # Median lines
        # =========================
        stats = sub.groupby("population_type")["ratio"].median().reindex(order)

        for k, pop in enumerate(stats.index):
            median = stats.loc[pop]

            ax.hlines(
                y=median,
                xmin=k-0.2,
                xmax=k+0.2,
                color=palette[pop],
                linewidth=2.5,
                zorder=2
            )

        # =========================
        # Clean axes
        # =========================
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelleft=True)
        ax.set_xlabel("")

        # =========================
        # X-axis control
        # =========================
        if i == 0:
            ax.set_xticks(range(len(order)))   # keep structure
            ax.set_xticklabels([])             # no labels
            ax.tick_params(axis='x', length=0) # no tick marks
        else:
            # BOTTOM ROW → keep structure but no labels
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([""] * len(order))

        # =========================
        # Add headroom
        y_max = sub["ratio"].max()
        y_min = sub["ratio"].min()
        
        # total range
        yrange = y_max - y_min
        
        # add padding to BOTH sides
        padding_top = yrange * 0.30
        padding_bottom = yrange * 0.10  # slightly less bottom padding looks nicer
        
        ax.set_ylim(y_min - padding_bottom, y_max + padding_top)
        ## =========================
        # Statistical annotation (per panel)
        # =========================
        x1, x2 = 0, 1
        
        y_max = sub["ratio"].max()
        y_min = sub["ratio"].min()
        yrange = y_max - y_min
        
        # position line ABOVE data
        y = y_max + yrange * 0.08
        h = yrange * 0.03
        
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
                lw=0.6, c='black', zorder=3)
        
        # optional: add text (e.g. "*", "ns", p-value)
        ax.text(
            (x1 + x2) * 0.5,
            y + h + yrange * 0.02,
            "",
            ha='center',
            va='bottom',
            fontsize=6,
            fontproperties=arial_font
        )

# =========================
# Bottom labels
# =========================
for ax in axes[-1, :]:
    ax.set_xlabel("")

# =========================
# Layout tuning
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.90,
    bottom=0.12,
    wspace=0.30,
    hspace=0.12
)

# =========================
# Save
# =========================
plt.savefig("Figure_ratio_4panels.svg", format="svg", dpi=300)
plt.show()
```

### 14.2. NS and SFS


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_context("talk")

ns = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/AF_outputs/NS_dataset_all.tsv", sep="\t")
sfs = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/AF_outputs/SFS_dataset_all.tsv", sep="\t")

print(ns.head())
print(sfs.head())

pop_order = ["PL","TR","SP","TM","RG","VS","MP","SC","BJ","KP","PK","PM","PJ","Island","Mainland", "Island_fieldwork","Mainland_fieldwork"]

ns["group"] = pd.Categorical(ns["group"], categories=pop_order, ordered=True)
sfs["group"] = pd.Categorical(sfs["group"], categories=pop_order, ordered=True)
```


```python
sfs.head()
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

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

sns.set_style("whitegrid")

# =========================
# Data prep
# =========================
bin_order_ns = ["0-5%", "5-10%", "10-20%", "20-80%", ">80%"]
bin_order_sfs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

ns_subset = ns.query("group in ['Mainland', 'Island']").copy()
sfs_subset = sfs.query("group in ['Mainland', 'Island']").copy()

groups = ["Mainland", "Island"]

colors = {"Mainland": "#208438", "Island": "#479BC9"}

gray_palette = {
    "SYN": "#d9d9d9",
    "DEL": "#969696",
    "LOF": "#252525"
}

# =========================
# Y limits
# =========================
def get_max(vals):
    vals = vals[~np.isnan(vals)]
    return np.max(vals) * 1.3

ns_ymax = get_max(ns_subset["N_S"].values)
sfs_ymax = get_max(sfs_subset["prop"].values)

# =========================
# Figure
# =========================
cm = 1 / 2.54
fig, axes = plt.subplots(
    2, 2,
    figsize=(9.53*cm, 7.51*cm),
    dpi=300,
    sharex=False,
    sharey='row'
)

# =========================
# Loop panels
# =========================
for i, row in enumerate(["NS", "SFS"]):
    for j, group in enumerate(groups):

        ax = axes[i, j]

        # Grid
        ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
        ax.set_axisbelow(True)

        # Spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

        # -----------------------
        # NS
        # -----------------------
        if row == "NS":

            data = ns_subset[ns_subset["group"] == group]

            sns.barplot(
                data=data,
                x="bin",
                y="N_S",
                order=bin_order_ns,
                color=colors[group],
                edgecolor="black",
                linewidth=0.4,
                ci=None,
                ax=ax
            )

            ax.set_xlim(-0.5, len(bin_order_ns)-0.5)
            ax.set_ylim(0, ns_ymax)

        # -----------------------
        # SFS
        # -----------------------
        else:

            data = sfs_subset[sfs_subset["group"] == group]

            sns.barplot(
                data=data,
                x="bin",
                y="prop",
                hue="category",
                order=bin_order_sfs,
                hue_order=["SYN", "DEL", "LOF"],
                palette=gray_palette,
                edgecolor="black",
                linewidth=0.4,
                ci=None,
                ax=ax
            )

            ax.set_xlim(-0.5, len(bin_order_sfs)-0.5)
            ax.set_ylim(0, sfs_ymax)

            if j != 1:
                ax.get_legend().remove()
            else:
                ax.legend(frameon=False, fontsize=6)

        # =========================
        # X-axis → ALWAYS ON
        # =========================
        ax.tick_params(axis='x', bottom=True, labelbottom=False, length=3, width=0.8)

        # =========================
        # Y-axis → ONLY LEFT COLUMN
        # =========================
        if j == 0:
            ax.tick_params(axis='y', left=True, labelleft=False, length=3, width=0.8)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

        # Clean labels
        ax.set_ylabel("")
        ax.set_xlabel("")

        # # Titles
        # if i == 0:
        #     ax.set_title(group, fontsize=8)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.90,
    bottom=0.12,
    wspace=0.12,
    hspace=0.30
)

# =========================
# Save
# =========================
plt.savefig("Figure_NS_SFS_final.svg", format="svg", dpi=300)
plt.show()
```

### 14.3. In ROHs vs. outside ROHs


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

# =========================
# Load Arial
# =========================
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
fm.fontManager.addfont(arial_path)
arial_font = fm.FontProperties(fname=arial_path)
arial_name = arial_font.get_name()

plt.rcParams['font.family'] = arial_name
plt.rcParams['font.sans-serif'] = [arial_name]

sns.set_style("whitegrid")

# =========================
# Load data
# =========================
df = pd.read_csv(
    "../results/population_analyses/rPodSic1.hap1.1/genetic_load/rohs/roh_load_summary.tsv",
    sep="\t"
)

# =========================
# Compute load ratios
# =========================
df["del_total"] = df["total_del"] / df["total_syn"]
df["lof_total"] = df["total_lof"] / df["total_syn"]

# =========================
# Reshape
# =========================
df_plot = pd.melt(
    df,
    id_vars=["sample", "region", "location"],
    value_vars=["del_total", "lof_total"],
    var_name="metric",
    value_name="ratio"
)

df_plot["metric"] = df_plot["metric"].map({
    "del_total": "DEL/SYN",
    "lof_total": "LOF/SYN"
})

# =========================
# Colors
# =========================
palette = {
    ("mainland", "in_roh"): "#208438",
    ("mainland", "out_roh"): "#145022",
    ("island", "in_roh"): "#479BC9",
    ("island", "out_roh"): "#255E7D"
}

order = ["mainland", "island"]
locations = ["in_roh", "out_roh"]

# =========================
# Figure
# =========================
cm = 1 / 2.54
fig, axes = plt.subplots(
    1, 2,
    figsize=(9.53 * cm, 4.5 * cm),
    dpi=300,
    sharex=True
)

metrics = ["DEL/SYN", "LOF/SYN"]

# =========================
# Plot
# =========================
for i, metric in enumerate(metrics):

    ax = axes[i]
    sub = df_plot[df_plot["metric"] == metric]

    # Grid
    ax.grid(axis='y', linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.grid(axis='x', linestyle="--", linewidth=0.5, color="white", alpha=0.7)
    ax.set_axisbelow(True)

    # Box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # -------------------------
    # Scatter + medians
    # -------------------------
    for j, region in enumerate(order):
        for loc in locations:

            vals = sub[
                (sub["region"] == region) &
                (sub["location"] == loc)
            ]["ratio"]

            if len(vals) == 0:
                continue

            offset = -0.2 if loc == "in_roh" else 0.2
            x = j + offset

            color = palette[(region, loc)]
            jitter = np.random.normal(0, 0.04, size=len(vals))

            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=14,
                alpha=0.5,
                color=color,
                edgecolors='none',
                linewidths=0
            )

            # median
            median = np.median(vals)
            ax.hlines(
                y=median,
                xmin=x - 0.15,
                xmax=x + 0.15,
                color=color,
                linewidth=2
            )

    # -------------------------
    # GLOBAL scaling for brackets
    # -------------------------
    all_vals = sub["ratio"]
    y_max_global = all_vals.max()
    y_min_global = all_vals.min()
    yrange_global = y_max_global - y_min_global

    # -------------------------
    # Statistical brackets (PROPER POSITIONING)
    # -------------------------
    for j, region in enumerate(order):
    
        x1 = j - 0.2
        x2 = j + 0.2
    
        vals1 = sub[(sub["region"] == region) & (sub["location"] == "in_roh")]["ratio"]
        vals2 = sub[(sub["region"] == region) & (sub["location"] == "out_roh")]["ratio"]
    
        if len(vals1) == 0 or len(vals2) == 0:
            continue
    
        local_vals = pd.concat([vals1, vals2])
        y_max_local = local_vals.max()
    
        # GLOBAL for scaling only
        h = yrange_global * 0.01
    
        if region == "mainland":
            # keep high (global)
            y = y_max_global + yrange_global * 0.08
        else:
            y = y_max_local + yrange_global * 0.08
    
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=0.6,
            c='black',
            zorder=3
        )
    
        ax.text(
            (x1 + x2) / 2,
            y + h + yrange_global * 0.015,
            "",
            ha='center',
            va='bottom',
            fontsize=6,
            fontproperties=arial_font
        )
    # -------------------------
    # Fix y-limits (avoid clipping)
    # -------------------------
    ax.set_ylim(
        y_min_global - yrange_global * 0.1,
        y_max_global + yrange_global * 0.3
    )

    # -------------------------
    # Remove ALL text
    # -------------------------
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    ax.tick_params(axis='x', bottom=True, labelbottom=False, length=3, width=0.8)
    ax.tick_params(axis='y', left=True, labelleft=False, length=3, width=0.8)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.90,
    bottom=0.12,
    wspace=0.30,
)

# =========================
# Save
# =========================
plt.savefig("Figure_total_load_ROH_clean.svg", format="svg", dpi=300)
plt.show()
```

### 14.4. ~Froh


```python
# Load your Excel file
df_long = pd.read_excel("df_long.xlsx")

# Optional: check it loaded correctly
df_long.head()
```


```python
# Load your Excel file
df_long_hom = pd.read_excel("df_long_hom.xlsx")

# Optional: check it loaded correctly
df_long_hom.head()
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

# =========================
# USER-DEFINED FIGURE SIZE
# =========================
width_cm = 9.53   # <-- change this
height_cm = 4.5   # <-- change this

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
# Figure layout (2 panels)
# =========================
cm = 1 / 2.54
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(width_cm*cm, height_cm*cm),
    dpi=300,
    sharex=True,
    sharey=False
)

# =========================
# Colors
# =========================
palette = {
    "mainland": "#208438",
    "island": "#479BC9"
}

regions = ["mainland", "island"]

# enforce order if needed
variants = ["hom_del", "hom_lof"]   # <-- safer than .unique()

# =========================
# Loop panels
# =========================
for i, variant in enumerate(variants):

    ax = axes[i]

    sub = df_long_hom[df_long_hom["variant_type"] == variant]

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.grid(axis="y", visible=True)

    # Plot both regions
    for region in regions:
        d = sub[sub["Region"] == region]

        # Scatter
        ax.scatter(
            d["Froh_centered"],
            d["hom"],
            s=10,
            alpha=0.5,
            color=palette[region],
            edgecolor="none",
            zorder=1
        )

        # Regression
        if len(d) > 1:
            x = d["Froh_centered"].values
            y = d["hom"].values

            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = np.polyval(coeffs, x_line)

            ax.plot(
                x_line,
                y_line,
                color=palette[region],
                linewidth=1.5,
                zorder=2
            )

    # Axes styling
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis='x', bottom=True, length=3, width=0.8, direction='out', pad=2)
    ax.tick_params(axis='y', left=True, length=3, width=0.8, direction='out', pad=2)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # Full box
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)

    # Panel title
    #ax.set_title(variant, fontsize=7)

# =========================
# Axis labels
# =========================
for ax in axes:
    ax.set_xlabel("")

axes[0].set_ylabel("")

# =========================
# Layout tuning
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.88,
    bottom=0.20,
    wspace=0.30
)

# =========================
# Save
# =========================
plt.savefig("Figure_regression_2panels.svg", format="svg", dpi=300)

plt.show()
```
