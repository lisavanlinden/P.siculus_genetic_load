# Analyses and plots population genomics


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

## 1. Principal Component Analysis

### 1.1. PCA plot


```python
# read the eigen vectors and values from the file
eigenvectors = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/PCA/psiculus_inbreeding_all.eigenvec", sep="\t").iloc[:, 2:].to_numpy()
eigenvalues = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/PCA/psiculus_inbreeding_all.eigenval", sep=" ", header=None).iloc[:, 0].to_numpy()
eigenvector_samples = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/PCA/psiculus_inbreeding_all.eigenvec", sep="\t", usecols=[1], header=None).iloc[:, 0].tolist()
```


```python
eigenvector_samples1 = eigenvector_samples[1:] 
```


```python
# extract the population codes from sample names
populations = [
    sample[2:4] if sample.startswith("24") else sample[:2] 
    for sample in eigenvector_samples
]
# remove "II"
populations = [pop for pop in populations if pop != 'II']
# recode CV and FV to RG
populations = ['RG' if pop in ['CV', 'FV'] else pop for pop in populations]
# get unique populations
unique_populations = list(set(populations))
unique_populations

```


```python
# define a specific color for each population
color_map = {
    'PJ':'#E4F1F8',
    'MP': '#C5E1EF',
    'KP': '#3A93C2',
    'PM': '#0C4A70',
    'BJ': '#6CB0D6',
    'SC': '#9EC9E2',
    'PK': '#246D9C',
    'VS': '#0A592A',
    'TM': '#3FAD5B',
    'RG': '#218B3B',
    'SP': '#CDE5D2',
    'TR': '#9CCEA7',
    'PL': '#6CB97D'
}
# assign colors to each sample based on population
population_colors = [color_map[pop] for pop in populations]

if len(population_colors) > eigenvectors.shape[0]:
    population_colors = population_colors[:eigenvectors.shape[0]]
```


```python
# calculate the proportion of variance explained by each PC
variance_explained = eigenvalues / np.sum(eigenvalues) * 100
```


```python
# create the PCA plot
plt.figure(figsize=(10, 10))
scatter = plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c=population_colors, edgecolor='#9B9FA1', s=500, linewidths = 0.5)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[pop], markersize=15) 
           for pop in unique_populations]
plt.legend(handles, unique_populations, loc="best", frameon=True, framealpha=0, edgecolor='black', fontsize = 18)
plt.axhline(0, color='#9B9FA1', linestyle='--', linewidth=0.5)
plt.axvline(0, color='#9B9FA1', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(f'PC1 ({variance_explained[0]:.2f}% explained var.)', fontsize = 25)
plt.ylabel(f'PC2 ({variance_explained[1]:.2f}% explained var.)', fontsize = 25)
plt.tight_layout()
#plt.savefig("1.svg", format="svg")
plt.show()
```

### 1.2. PCA plot islands only


```python
# read the eigen vectors and values from the file
eigenvectors = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/PCA/psiculus_inbreeding_island.eigenvec", sep="\t").iloc[:, 2:].to_numpy()
eigenvalues = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/PCA/psiculus_inbreeding_island.eigenval", sep=" ", header=None).iloc[:, 0].to_numpy()
eigenvector_samples = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/PCA/psiculus_inbreeding_island.eigenvec", sep="\t", usecols=[1], header=None).iloc[:, 0].tolist()
```


```python
# extract the population codes from sample names
populations = populations = [
    sample[2:4] if sample.startswith("24") else sample[:2] 
    for sample in eigenvector_samples]

populations = [pop for pop in populations if pop != 'II']
unique_populations = list(set(populations))
unique_populations
```


```python
# define a specific color for each population

color_map = {
    'PJ':'#E4F1F8',
    'MP': '#C5E1EF',
    'KP': '#3A93C2',
    'PM': '#0C4A70',
    'BJ': '#6CB0D6',
    'SC': '#9EC9E2',
    'PK': '#246D9C',
    'VS': '#0A592A',
    'TM': '#3FAD5B',
    'RG': '#218B3B',
    'SP': '#CDE5D2',
    'TR': '#9CCEA7',
    'PL': '#6CB97D'
}
# assign colors to each sample based on population
population_colors = [color_map[pop] for pop in populations]

if len(population_colors) > eigenvectors.shape[0]:
    population_colors = population_colors[:eigenvectors.shape[0]]
```


```python
# calculate the proportion of variance explained by each PC
variance_explained = eigenvalues / np.sum(eigenvalues) * 100
```


```python
# create the PCA plot
plt.figure(figsize=(10, 10))
scatter = plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c=population_colors, edgecolor='#9B9FA1', s=500, linewidths = 0.5)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[pop], markersize=15) 
           for pop in unique_populations]
plt.legend(handles, unique_populations, loc="best", frameon=True, framealpha=0, edgecolor='black', fontsize = 18)
plt.axhline(0, color='#9B9FA1', linestyle='--', linewidth=0.5)
plt.axvline(0, color='#9B9FA1', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(f'PC1 ({variance_explained[0]:.2f}% explained var.)', fontsize = 25)
plt.ylabel(f'PC2 ({variance_explained[1]:.2f}% explained var.)', fontsize = 25)
plt.tight_layout()
#plt.savefig("1.svg", format="svg")
plt.show()
```

## 3. ADMIXTURE

### 3.0. CV error


```python
# cross-validation error result: K = ...
cv = "../results/population_analyses/rPodSic1.hap1.1/admixture/psiculus_inbreeding_admixture2.cv.error"
cv_error = pd.read_csv(cv, sep=' ', header=None)
cv_error
```

### 3.1. Admixture plot with best K-value


```python
# full path to the PED file used for Admixture
fileped = "../results/population_analyses/rPodSic1.hap1.1/admixture/psiculus_inbreeding_admixture2_columns.ped"
# full path to the Q file to use for plotting
fileq = "../results/population_analyses/rPodSic1.hap1.1/admixture/psiculus_inbreeding_admixture2.2.Q"
# read in ped file as dataframe, this file is delimited by whitespace
df_ped = pd.read_csv(fileped, sep=' ', header=None)
# read in the Q file, which is also whitespace delimited 
df_q = pd.read_csv(fileq, sep=' ', header=None)
```


```python
# automatically generate column names based on number of columns in Q (pop1, pop2, pop3, etc.)
names = ["pop{}".format(i) for i in range(1, df_q.shape[1]+1)]
# add column names to dataframe
df_q.columns = names
# insert the sample names into the first column position
df_q.insert(0, 'Sample', df_ped[0])
# now  set the dataframe index to the sample names (e.g., the 'Sample' column)
df_q.set_index('Sample', inplace=True)

df_q['assignment'] = df_q.idxmax(axis=1)
```


```python
pal = sns.color_palette(['#ADD8E6',"#a6daae"])
pal
```


```python
def sort_df_by_pops_nocat(df):
    temp_dfs = []
    for pop in sorted(df['assignment'].unique()):
        temp = df.loc[df['assignment'] == pop].sort_values(by=[pop], ascending=False)
        temp_dfs.append(temp)
    return temp_dfs
```


```python
# A list of the subdataframes is returned:
sub_dfs = sort_df_by_pops_nocat(df_q)
df_custom_sort = pd.concat([sub_dfs[0], sub_dfs[1]])

# Remember python indices are zero-based!
# If your K value is higher or lower, make sure to include all the subdataframes in 
# the list for pd.concat() . For example with K=6, you'll need sub_dfs[0] to sub_dfs[5].
```


```python
ax = df_custom_sort.plot.bar(stacked=True, 
                             figsize=(15,5), 
                             width=1,
                             color=pal, 
                             fontsize='x-small',
                             edgecolor='black', 
                             linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticklabels(df_custom_sort.index, rotation=90, ha='right')
ax.legend(bbox_to_anchor=(1,1), fontsize='medium', labelspacing=0.5, frameon=False)
```

### 3.2. Admixture plot with all K-values


```python
# File paths
ped_file = "../results/population_analyses/rPodSic1.hap1.1/admixture/psiculus_inbreeding_admixture2_columns.ped"
metadata_file = "../metadata/samples2025_pop.txt"
K_values = 2,3,4,7,8

# Load .ped file to get sample order
df_ped = pd.read_csv(ped_file, sep=' ', header=None)
df_ped.columns = ['FamilyID', 'SampleID'] + [f"col{i}" for i in range(2, df_ped.shape[1])]
sample_order = df_ped['SampleID'].tolist()

# Load population metadata
pop_df = pd.read_csv(metadata_file, sep='\t', header=None, names=["Sample", "Population"])

# Reassign CV* and FV* samples to RG
pop_df.loc[pop_df["Sample"].str.startswith(("CV", "FV")), "Population"] = "RG"

# Keep only samples in sample_order and preserve that order
pop_df = pop_df[pop_df["Sample"].isin(sample_order)]
pop_df = pop_df.set_index("Sample").loc[sample_order].reset_index()

# Define your custom population order
desired_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]
pop_df["Population"] = pd.Categorical(pop_df["Population"], categories=desired_order, ordered=True)
```


```python
desired_order
```


```python
# Generate a color palette generator function
def get_colors(n):
    """
    Generate n visually distinct colors from a custom 8-color palette.
    """
    # Define the 8-color palette
    palette = [
    "#8A7661",   # deep muted brown, more rich
    "#D3D3D3",  # light beige  #ACA49B
    "#D9CFC0",  # soft taupe
    "#C8B39D",  # muted tan, warmer
    "#5C4532",  # warm grey-brown, more saturated
    "#EDE8DF",  # darker earthy brown, more contrast
    "#A07F66",  # cool grey, slightly warmer
    "#B69078",  # very light, slightly warmer
]
    
    # If n > palette length, interpolate by repeating colors
    if n <= len(palette):
        return palette[:n]
    else:
        # Repeat colors evenly
        idxs = np.linspace(0, len(palette) - 1, n)
        colors = [palette[int(round(i))] for i in idxs]
        return colors
    
# Function to plot a single admixture barplot
# Load Arial from the specific path
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"  # adjust path if needed
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()  # set globally

# Function to plot a single admixture barplot
def plot_admixture(q_df, k, ax, colors):
    ancestry_cols = [col for col in q_df.columns if col.startswith("Ancestry_")]
    
    # Plot stacked bars
    q_df[ancestry_cols].plot(kind='bar', stacked=True, ax=ax, width=1.0,
                             linewidth=0.3, edgecolor='black', color=colors, legend=False)

    # Remove extra space around bars
    ax.margins(x=0)
    ax.set_xlim(-0.5, len(q_df)-0.5)
    ax.set_ylim(0, 1)

    # Add vertical lines between populations
    pop_change = q_df["Population"].ne(q_df["Population"].shift()).cumsum()
    boundaries = q_df.groupby(pop_change, observed=True).size().cumsum()[:-1]
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color='black', linestyle='--', linewidth=1)

    # Add population labels
    pop_sizes = q_df.groupby("Population", observed=True).size()
    pop_centers = pop_sizes.cumsum() - pop_sizes / 2
    for pop, center in pop_centers.items():
        ax.text(center, -0.1, pop, ha="center", va="top", rotation=45,
                fontsize=6, fontproperties=arial_font, transform=ax.get_xaxis_transform())

    ax.set_title(f"K={k}", fontsize=10, fontproperties=arial_font)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('', fontproperties=arial_font)
    ax.set_ylabel('', fontsize=8, fontproperties=arial_font)

```


```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# # Load Arial font
# arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"  # adjust path if needed
# arial_font = fm.FontProperties(fname=arial_path)
# plt.rcParams['font.family'] = arial_font.get_name()

# Convert cm to inches for figsize
cm = 1 / 2.54
fig = plt.figure(figsize=(18*cm, 12*cm))  # width x height in inches
gs = GridSpec(len(K_values), 1, figure=fig)

last_q_df = None
last_colors = None

for i, k in enumerate(K_values):
    ax = fig.add_subplot(gs[i, 0])
    q_file = f"../results/population_analyses/rPodSic1.hap1.1/admixture/psiculus_inbreeding_admixture2.{k}.Q"
    df_q = pd.read_csv(q_file, delim_whitespace=True, header=None)
    df_q.columns = [f"Ancestry_{j+1}" for j in range(k)]
    df_q["Sample"] = pop_df["Sample"].values
    df_q["Population"] = pop_df["Population"].values
    df_q = df_q.sort_values(by=["Population", "Sample"]).reset_index(drop=True)

    colors = get_colors(k)
    plot_admixture(df_q, k, ax, colors)

    # Save final for global legend
    last_q_df = df_q
    last_colors = colors

# Global legend using last_q_df
# ancestry_cols = [col for col in last_q_df.columns if col.startswith("Ancestry_")]
# handles = [plt.Rectangle((0, 0), 1, 1, color=last_colors[i]) for i in range(len(ancestry_cols))]
# fig.legend(handles, ancestry_cols, loc='upper right', bbox_to_anchor=(1.15, 0.9), fontsize=9)

plt.tight_layout()
#plt.savefig("Figure_1c_v1.png", format="png", dpi=300, bbox_inches='tight')
#plt.savefig("Figure_1c_v1_test.svg", format="svg", dpi=300, bbox_inches='tight')
plt.show()

```

## 4. Fst


```python
# Genome-wide output mainland vs. island
    # Weir and Cockerham mean Fst estimate: 0.086975
    # Weir and Cockerham weighted Fst estimate: 0.41471

# Genome-wide output clusters Lastovo vs. PJ
    # Weir and Cockerham mean Fst estimate: 0.26804
    # Weir and Cockerham weighted Fst estimate: 0.32685

# Genome-wide output clusters Lastovo vs. MP
    # Weir and Cockerham mean Fst estimate: 0.13984
    # Weir and Cockerham weighted Fst estimate: 0.23084

# Genome-wide output clusters MP vs. PJ
    # Weir and Cockerham mean Fst estimate: 0.19384
    # Weir and Cockerham weighted Fst estimate: 0.32304
```


```python
# Read data from the output file of vcftools
Fst_data = "../results/population_analyses/rPodSic1.hap1.1/Fst/psiculus_inbreeding_mainland_island_50kb_Fst.windowed.weir.fst"
# Use pandas to read the file
Fst = pd.read_csv(Fst_data, sep='\t', skiprows=1, names=['CHROM', 'BIN_START', 'BIN_END', 'N_VARIANTS', 'WEIGHTED_FST', 'MEAN_FST'])
```


```python
# Add a new column for the x-axis index
Fst['chromosome'] = Fst['CHROM'].astype('category')
Fst['chromosome'] = Fst['chromosome'].cat.set_categories(sorted(Fst['chromosome'].unique()), ordered=True)
Fst = Fst.sort_values(['chromosome', 'BIN_START'])
Fst['ind'] = range(len(Fst))

# Group data by chromosome
Fst_grouped = Fst.groupby('chromosome', observed=True)

# Calculate the 99.5th percentile for the FST values
percentile_99_5 = Fst['MEAN_FST'].quantile(0.995)
print(percentile_99_5)

# Compute the genome-wide average weighted FST
weighted_fst_avg = Fst['WEIGHTED_FST'].mean()
print(weighted_fst_avg)

# Compute the genome-wide average mean FST
mean_fst_avg = Fst['MEAN_FST'].mean()
print(mean_fst_avg)
```


```python
# Create the Manhattan plot
fig = plt.figure(figsize=(14, 8))  # Set figure size
ax = fig.add_subplot(111)
colors = ['grey', 'black']  # Define alternating colors
x_labels = []
x_labels_pos = []

for num, (name, group) in enumerate(Fst_grouped):
    group.plot(kind='scatter', x='ind', y='MEAN_FST', color=colors[num % len(colors)], ax=ax, s=10)
    x_labels.append(name)
    x_labels_pos.append((group['ind'].iloc[-1] + group['ind'].iloc[0]) / 2)

# Add 99.5th percentile threshold line
ax.axhline(y=percentile_99_5, color='red', linestyle='-', label='99.5th Percentile')
ax.axhline(y=weighted_fst_avg, color='blue', linestyle='--', label='Weighted FST Average')
ax.axhline(y=mean_fst_avg, color='green', linestyle='--', label='Mean FST Average')
# Set x-axis labels
ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Set axis limits
ax.set_xlim([0, len(Fst)])
ax.set_ylim([Fst['MEAN_FST'].min(), Fst['MEAN_FST'].max()])

# Add labels and title
ax.set_xlabel('Chromosome')
ax.set_ylabel('Mean FST')
ax.set_title('Fst-scan')

# Show the plot
plt.legend()
plt.tight_layout()
plt.show()
```

## 6. BCFtools/RoH 

### 6.1. BCFtools/RoH Viterbi training


```python
# Step 1: Load the Viterbi output
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/bcftools_roh/vt_final_per_sample.txt", sep='\s+', header=None)
df.columns = ["type", "sample", "iteration", "dAZ", "dHW", "1-P(HW|HW)", "P(AZ|HW)", "1-P(AZ|AZ)", "P(HW|AZ)"]

# Keep only relevant columns
vt = df[["sample", "P(AZ|HW)", "P(HW|AZ)"]].copy()

# ──────────────────────────────────────────────────────────────
# Step 2: Extract population codes from sample names
populations = []
for s in vt["sample"]:
    if s.startswith("24"):
        pop = s[2:4]
    else:
        pop = s[:2]
    populations.append(pop)

# Clean and recode populations
populations = ["RG" if p in ["CV", "FV"] else p for p in populations]
vt["population"] = populations

# ──────────────────────────────────────────────────────────────
# Step 3: Define regions and assign each population
mainland_pops = ["PL", "TR", "SP", "TM", "RG", "VS"]
island_pops   = ["MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]

def assign_region(pop):
    if pop in mainland_pops:
        return "Mainland"
    elif pop in island_pops:
        return "Island"
    else:
        return "Unknown"

vt["region"] = vt["population"].apply(assign_region)

# ──────────────────────────────────────────────────────────────
# Step 4: Summaries

# (a) Per-population
vt_pop = (
    vt.groupby("population")[["P(AZ|HW)", "P(HW|AZ)"]]
      .agg(["mean", "std", "count"])
      .reset_index()
)

# (b) Per-region (Mainland vs Island)
vt_region = (
    vt.groupby("region")[["P(AZ|HW)", "P(HW|AZ)"]]
      .agg(["mean", "std", "count"])
      .reset_index()
)

# (c) Overall
vt_overall = vt[["P(AZ|HW)", "P(HW|AZ)"]].agg(["mean", "std", "count"])

# ──────────────────────────────────────────────────────────────
# Step 5: Print summaries
print("🧬 Per-population summary:\n")
print(vt_pop, "\n")

print("🌍 Mainland vs Island summary:\n")
print(vt_region, "\n")

print("📊 Overall summary:\n")
print(vt_overall, "\n")

# ──────────────────────────────────────────────────────────────
# Step 6: Save summaries to files
#vt_pop.to_csv("vt_per_population_summary.txt", sep="\t", index=False)
#vt_region.to_csv("vt_per_region_summary.txt", sep="\t", index=False)
#vt_overall.to_csv("vt_overall_summary.txt", sep="\t", index=True)

```

### 6.2. BCFtools/RoH summary


```python
#roh_file = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/bcftools_roh/bcftools_roh_r_out.txt' # version 1
roh_file = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/bcftools_roh/bcftools_roh_r_out_v2.txt' # version 2 = good one!

df_bcftoolsROH = pd.read_csv(
    roh_file,
    comment='#',  # skip bcftools headers
    sep='\s+',
    header=None,
    names=["Tag", "Sample", "Chromosome", "Start", "End", "Length_bp", "Markers", "Quality"]
)

df_bcftoolsROH = df_bcftoolsROH[df_bcftoolsROH["Tag"] == "RG"]
print(df_bcftoolsROH)
```


```python
# Extract population code from Individual_ID
df_bcftoolsROH['Population'] = df_bcftoolsROH['Sample'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))

# Step 1: collapse "24XX" into "XX"
df_bcftoolsROH['Population'] = df_bcftoolsROH['Population'].str.replace(r'^24', '', regex=True)

# Step 2: merge CV and FV into RG
df_bcftoolsROH['Population'] = df_bcftoolsROH['Population'].replace({'CV': 'RG', 'FV': 'RG'})
df_bcftoolsROH.head()

```

#### 6.2.1. %ROH plot


```python
# roh length per individual
roh_per_ind = df_bcftoolsROH.groupby(["Population", "Sample"])["Length_bp"].sum().reset_index()
roh_per_ind.rename(columns={"Length_bp": "total_roh_bp"}, inplace=True)

# genome size

fai_file = "../genome/GCA_964188175.1_rPodSic1.hap1.1_genomic.fna.fai" 
fai = pd.read_csv(fai_file, sep="\t", header=None, usecols=[0, 1], names=["Chromosome", "Length"])

scaffolds_in_roh = set(df_bcftoolsROH["Chromosome"].unique())

missing_from_fai = scaffolds_in_roh - set(fai["Chromosome"])
if missing_from_fai:
    print("Warning: these scaffolds in the ROH file were NOT found in the .fai (check naming):")
    print(missing_from_fai)

# Keep only scaffolds that are both in .fai and in the ROH/VCF (this matches autosomes used)
fai_sub = fai[fai["Chromosome"].isin(scaffolds_in_roh)].copy()

genome_length_autosomes = fai_sub["Length"].sum()
print(f"Genome length (sum of scaffolds present in ROH/VCF): {genome_length_autosomes:,} bp")

# % of genome in roh

roh_per_ind["percent_genome_in_roh"] = roh_per_ind["total_roh_bp"] / genome_length_autosomes * 100
roh_per_ind["Froh"] = roh_per_ind["percent_genome_in_roh"] / 100
# === 5. Save summary ===
#roh_per_ind.to_csv("bcftools_roh_summary.csv", index=False)
#print(roh_per_ind)
roh_per_ind.head()
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# =========================
# Load and register Arial
# =========================
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"

fm.fontManager.addfont(arial_path)
arial_font = fm.FontProperties(fname=arial_path)
arial_name = arial_font.get_name()

plt.rcParams['font.family'] = arial_name
plt.rcParams['font.sans-serif'] = [arial_name]

# Font sizes (same system as other figures)
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


# =========================
# Colors and order
# =========================
color_map = {
    'PJ':'#E4F1F8','MP': '#C5E1EF','KP': '#3A93C2','PM': '#0C4A70',
    'BJ': '#6CB0D6','SC': '#9EC9E2','PK': '#246D9C','VS': '#0A592A',
    'TM': '#3FAD5B','RG': '#218B3B','SP': '#CDE5D2','TR': '#9CCEA7','PL': '#6CB97D'
}

desired_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]


# =========================
# Figure size
# =========================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(12.3*cm, 6*cm), dpi=300)


# =========================
# Grid
# =========================
ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

#ax.minorticks_on()

ax.grid(
    which='minor',
    axis='y',
    linestyle='--',
    linewidth=0.3,
    color='grey',
    alpha=0.5
)


# =========================
# Strip plot
# =========================
sns.stripplot(
    data=roh_per_ind,
    x='Population',
    y='Froh',
    palette=color_map,
    jitter=True,
    size=8,
    order=desired_order,
    alpha=0.5,
    zorder=1,
    ax=ax
)

ax.xaxis.set_tick_params(which='minor', bottom=False)


# =========================
# Median lines
# =========================
stats = roh_per_ind.groupby('Population')['Froh'].median().reindex(desired_order)

for i, pop in enumerate(desired_order):

    ax.hlines(
        y=stats.loc[pop],
        xmin=i-0.3,
        xmax=i+0.3,
        color=color_map[pop],
        linewidth=3,
        zorder=2
    )


# =========================
# Labels
# =========================
ax.set_ylabel("", labelpad=10)
ax.set_xlabel("")

# keep ticks but hide population labels
ax.tick_params(axis='x', labelbottom=False, pad=4)
#ax.tick_params(axis='y', labelleft=False,pad=4)


# =========================
# Layout (same system)
# =========================
ax.set_position([0.10, 0.18, 0.88, 0.74])


# =========================
# Save
# =========================
fig.canvas.draw()

plt.savefig(
    "Figure_3c_v1.svg",
    format="svg",
    dpi=300
)

plt.show()
```


```python

```

#### 6.2.2. ROH sizes


```python
# --- Create ROH length bins ---
#bins = [0, 5e4, 1e5, 5e5, 1e6, 3e6, 3e7] 
#labels = ['<50Kb', '50-100Kb', '100-500Kb', '500Kb-1Mb', '1-3Mb', '>3Mb']

bins = [5e4, 1e5, 5e5, 1e6, 3e6, 3e9]  
labels = ['50-100Kb', '100-500Kb', '500Kb-1Mb', '1-3Mb', '>3Mb']
df_bcftoolsROH['ROH_bin'] = pd.cut(df_bcftoolsROH['Length_bp'], bins=bins, labels=labels, right=False)
print(df_bcftoolsROH)
```


```python
# --- Create ROH length bins ---
bins = [1e4, 5e4, 1e5, 5e5, 1e6, 1e9]
labels = [
    '10-50Kb',
    '50-100Kb',
    '100-500Kb',
    '500Kb-1Mb',
    '>1Mb',
]

df_bcftoolsROH['ROH_bin'] = pd.cut(
    df_bcftoolsROH['Length_bp'],
    bins=bins,
    labels=labels,
    right=False
)

```


```python
# === Calculate average number of ROHs per bin per population ===
roh_counts = (
    df_bcftoolsROH.groupby(['Population', 'Sample', 'ROH_bin'], observed=True)
      .size()
      .reset_index(name='Count')
)

avg_roh_counts = (
    roh_counts.groupby(['Population', 'ROH_bin'], observed=True)['Count']
    .mean()
    .reset_index()
)
```


```python
mainland_pops = ["PL", "TR", "SP", "TM", "RG", "VS"]
island_pops   = ["MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]

def assign_region(pop):
    if pop in mainland_pops:
        return "Mainland"
    elif pop in island_pops:
        return "Island"
    else:
        return "Unknown"

roh_counts['Region'] = roh_counts['Population'].apply(assign_region)
roh_counts = roh_counts[~roh_counts['Sample'].isin(['PMU01'])]
#df_angsd.to_excel("ANGSD_summary.xlsx", index=False)

from scipy.stats import mannwhitneyu

results = []

for b in roh_counts['ROH_bin'].dropna().unique():
    
    subset = roh_counts[roh_counts['ROH_bin'] == b]
    
    ml = subset[subset['Region'] == 'Mainland']['Count']
    il = subset[subset['Region'] == 'Island']['Count']
    
    stat, p = mannwhitneyu(ml, il, alternative='two-sided')
    
    results.append({
        'ROH_bin': b,
        'ML_mean': ml.mean(),
        'IL_mean': il.mean(),
        'p_value': p
    })

import pandas as pd
pd.DataFrame(results)
```


```python
import pandas as pd

roh_counts['Region'] = roh_counts['Region'].astype('category')
roh_counts['ROH_bin'] = roh_counts['ROH_bin'].astype('category')
roh_counts['Population'] = roh_counts['Population'].astype('category')

import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.gee(
    "Count ~ Region * ROH_bin",
    groups="Population",
    data=roh_counts,
    family=sm.families.Poisson()
)

result = model.fit()

print(result.summary())
```


```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

# === Order of populations ===
desired_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]

# === Figure setup (THREE PANELS = TWO BREAKS) ===
cm = 1 / 2.54
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True,
    figsize=(25 * cm, 10 * cm),
    gridspec_kw={'height_ratios': [0.5, 0.5, 2], 'hspace': 0.05}
)

# === Define y-axis ranges ===
ymin = 0
ybreak1 = 200
ybreak2 = 5000
ymax = 170000

# === Shared barplot settings ===
common_kws = dict(
    data=avg_roh_counts,
    x='Population',
    y='Count',
    hue='ROH_bin',
    order=desired_order,
    palette='Greys',
    edgecolor='black',
    linewidth=0.4,
    alpha=1
)

# === Plot bars on all panels ===
sns.barplot(**common_kws, ax=ax1)
sns.barplot(**common_kws, ax=ax2)
sns.barplot(**common_kws, ax=ax3)

# === Apply y-limits ===
ax1.set_ylim(ybreak2, ymax)
ax2.set_ylim(ybreak1, ybreak2)
ax3.set_ylim(ymin, ybreak1)

# === Hide spines between panels ===
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)

# === Tick handling ===
ax1.tick_params(labelbottom=False, bottom=False)
ax2.tick_params(labelbottom=False, bottom=False)
ax3.tick_params(axis='x', labelbottom=False, bottom=True)

# === Add diagonal break marks ===
d = 0.5
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=6,
    linestyle="none",
    color='k',
    mec='k',
    mew=1,
    clip_on=False
)

# Break between ax1 and ax2
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# Break between ax2 and ax3
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)

# === Y-axis label (only bottom panel) ===
ax1.set_ylabel("")
ax2.set_ylabel("")
ax3.set_ylabel(
    "Average number of ROHs",
    fontsize=8,
    fontproperties=arial_font,
    labelpad=15
)
ax3.yaxis.set_label_coords(-0.05, 0.65)

# === Grid + styling ===
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("")
    ax.tick_params(labelsize=6)
    ax.yaxis.label.set_fontproperties(arial_font)
    ax.set_axisbelow(True)

    ax.grid(True, which='both', linestyle="--",
            linewidth=0.4, color="grey", alpha=0.6, axis='y')
    ax.grid(True, which='minor', linestyle="--",
            linewidth=0.2, color="grey", alpha=0.3, axis='y')

# === Legend ===
ax1.legend_.remove()
ax2.legend_.remove()
leg = ax3.legend(
    title="",
    fontsize=6,
    title_fontsize=1,
    loc='center left',
    bbox_to_anchor=(1.02, 0.70),
    frameon=False,
)

# === Layout ===
plt.tight_layout(h_pad=0.01, rect=[0, 0, 0.9, 1])
# plt.savefig("avg_roh_bins_per_population.png", dpi=300)
plt.show()

```


```python
import pandas as pd

# --- Define genome length ---
genome_length = 1423562263

# --- Define ROH thresholds ---
thresholds = [50_000, 100_000, 500_000, 1_000_000, 2_000_000, 3_000_000]

# --- Initialize dictionary to store per-individual sums ---
roh_dict = {}

for t in thresholds:
    col_name = f'Sum_ROH_gt{int(t/1000)}kb'  # Column name like Sum_ROH_gt50kb
    temp = (
        df_bcftoolsROH[df_bcftoolsROH['Length_bp'] >= t]
        .groupby('Sample')['Length_bp']
        .sum()
        .reset_index(name=col_name)
    )
    if not roh_dict:
        roh_dict['df'] = temp
    else:
        roh_dict['df'] = roh_dict['df'].merge(temp, on='Sample', how='outer')

# --- Final dataframe ---
df_Froh_sizes = roh_dict['df']

# --- Calculate FROH for each threshold ---
for t in thresholds:
    col_sum = f'Sum_ROH_gt{int(t/1000)}kb'
    col_froh = f'FROH_gt{int(t/1000)}kb'
    df_Froh_sizes[col_froh] = df_Froh_sizes[col_sum] / genome_length

# --- Optional: fill NaN with 0 for individuals with no ROHs in that category ---
df_Froh_sizes = df_Froh_sizes.fillna(0)

print(df_Froh_sizes)

```


```python
# Extract population code from Individual_ID
df_Froh_sizes['Population'] = df_Froh_sizes['Sample'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))

# Step 1: collapse "24XX" into "XX"
df_Froh_sizes['Population'] = df_Froh_sizes['Population'].str.replace(r'^24', '', regex=True)

# Step 2: merge CV and FV into RG
df_Froh_sizes['Population'] = df_Froh_sizes['Population'].replace({'CV': 'RG', 'FV': 'RG'})

mainland_pops = ["PL", "TR", "SP", "TM", "RG", "VS"]
island_pops   = ["MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]

def assign_region(pop):
    if pop in mainland_pops:
        return "Mainland"
    elif pop in island_pops:
        return "Island"
    else:
        return "Unknown"

df_Froh_sizes['Region'] = df_Froh_sizes['Population'].apply(assign_region)
print(df_Froh_sizes)
df_Froh_sizes.to_excel("Froh_sizes_summary.xlsx", index=False)

```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import pandas as pd

# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

# === Define color map and population order ===
color_map = {
    'PJ':'#6BAED6',
    'MP': '#2171B5',
    'KP': '#4292C6',
    'PM': '#08306B',
    'BJ': '#08519C',
    'SC': '#9ECAE1',
    'PK': '#C6DBEF',
    'VS': '#238B45',
    'TM': '#00441B',
    'RG': '#BAE4B3',
    'SP': '#74C476',
    'TR': '#006D2C',
    'PL': '#41AB5D'
}
desired_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]

# === Define thresholds for which Froh was calculated ===
thresholds = [50_000, 100_000, 500_000, 1_000_000, 2_000_000, 3_000_000]

# === Loop through each FROH threshold and make a plot ===
cm = 1 / 2.54  # cm to inches

for t in thresholds:
    col_froh = f'FROH_gt{int(t/1000)}kb'
    
    plt.figure(figsize=(20*cm, 8*cm))
    plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    plt.minorticks_on()
    plt.grid(which='minor', axis='y', linestyle='--', linewidth=0.3, color='grey', alpha=0.5)
    
    ax = sns.stripplot(
        data=df_Froh_sizes,
        x='Population',
        y=col_froh,
        palette=color_map,
        jitter=True,
        size=6,
        order=desired_order,
        alpha=0.3,
        zorder=1
    )
    ax.xaxis.set_tick_params(which='minor', bottom=False)

    # Calculate median per population
    stats = df_Froh_sizes.groupby('Population')[col_froh].median().reindex(desired_order)

    # Add median lines
    for i, pop in enumerate(desired_order):
        median = stats.loc[pop]
        color = color_map[pop]
        ax.hlines(
            y=median,
            xmin=i - 0.3,
            xmax=i + 0.3,
            color=color,
            linewidth=3,
            zorder=2
        )

    # === Styling ===
    ax.set_ylabel(f"{col_froh.replace('_', ' ')}", fontsize=8, fontproperties=arial_font)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    # === Save and show ===
    plt.savefig(f"Figure_Froh_{int(t/1000)}kb.png", format="png", dpi=300)
    plt.show()

```

## 7. ANGSD

### 7.1. ANGSD H summary per individual


```python
# set your folder path
angsd_path = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/angsd/'
```


```python
import pandas as pd
import glob
import os
import numpy as np

# Path to your folder
angsd_path = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/angsd/'

# Get all individuals that have the global.het.txt file
global_files = glob.glob(os.path.join(angsd_path, '*_global.het.txt'))

summary_data = []

for gfile in global_files:
    indiv = os.path.basename(gfile).split('_global')[0]  # e.g. "VS10"
    # --- Read global heterozygosity ---
    try:
        with open(gfile) as f:
            global_h = float(f.readline().split()[0])
    except Exception as e:
        print(f"Error reading {gfile}: {e}")
        global_h = np.nan

    # --- Read 100KB heterozygosity file and compute mean ---
    hfile = os.path.join(angsd_path, f"{indiv}_100KB.het.txt")
    if os.path.exists(hfile):
        try:
            df = pd.read_csv(hfile, sep='\s+', header=None)
            mean_h = df[0].mean()
        except Exception as e:
            print(f"Error reading {hfile}: {e}")
            mean_h = np.nan
    else:
        mean_h = np.nan

    summary_data.append([indiv, global_h, mean_h])

# Create dataframe
df_angsd = pd.DataFrame(summary_data, columns=['individual', 'global_het', 'mean_100KB_het'])
df_angsd.head()
```


```python
# First, create Population column, extracting letters from Individual_ID
df_angsd['population'] = df_angsd['individual'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
# Reassign CV and FV to RG
df_angsd.loc[df_angsd['population'].isin(['CV', 'FV']), 'population'] = 'RG'
# Define custom population order
pop_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]
```


```python
mainland_pops = ["PL", "TR", "SP", "TM", "RG", "VS"]
island_pops   = ["MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]

def assign_region(pop):
    if pop in mainland_pops:
        return "Mainland"
    elif pop in island_pops:
        return "Island"
    else:
        return "Unknown"

df_angsd['region'] = df_angsd['population'].apply(assign_region)
df_angsd = df_angsd[~df_angsd['individual'].isin(['PMU01'])]
#df_angsd.to_excel("ANGSD_summary.xlsx", index=False)
```

### 7.2. Heterozygosity plot per type


```python
import statsmodels.formula.api as smf
# Fit a mixed-effects model: Region as fixed effect, Population as random effect
model = smf.mixedlm("mean_100KB_het ~ region", df_angsd, groups=df_angsd["population"])
result = model.fit()

print(result.summary())
```


```python
model = smf.ols("mean_100KB_het ~ region", data=df_angsd).fit()
print(model.summary())
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

# === Figure size in cm (~10x10 cm) ===
cm = 1 / 2.54
plt.figure(figsize=(8*cm, 8*cm))

# === Define colors ===
palette = {"Mainland": "#009900", "Island": "#0066CC"}

# --- Grid setup ---
plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)  # major grid
plt.minorticks_on()
plt.grid(
    which='minor',
    axis='y',   # horizontal minor lines only
    linestyle='--',
    linewidth=0.3,
    color='grey',
    alpha=0.5
)

# --- Strip plot (dots in background) ---
ax = sns.stripplot(
    data=df_angsd,
    x='region',
    y='mean_100KB_het',
    hue='region',
    palette=palette,
    jitter=True,
    size=6,
    alpha=0.3,  # slightly transparent dots
    zorder=1,    # behind median lines
    dodge=False
)
ax.xaxis.set_tick_params(which='minor', bottom=False)  # remove minor x ticks

# --- Add median lines per Region ---
stats = df_angsd.groupby('region')['mean_100KB_het'].median().reindex(palette.keys())

for i, region in enumerate(stats.index):
    median = stats.loc[region]
    color = palette[region]
    
    ax.hlines(
        y=median,
        xmin=i - 0.2,
        xmax=i + 0.2,
        color=color,
        linewidth=3,
        zorder=2  # above dots
    )

# --- Axes labels and ticks ---
ax.set_ylabel("Heterozygosity", fontsize=8, fontproperties=arial_font)
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=45)
plt.xticks(ticks=range(len(stats.index)), labels=[""]*len(stats.index), fontsize=6)
plt.yticks(fontsize=6)

# --- Optional: Statistical annotation ---
x1, x2 = 0, 1
y_max = df_angsd['mean_100KB_het'].max()
y = y_max + 0.0005  # start just above max value
h = 0.00005          # height of short vertical ticks
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c='black')
ax.text((x1+x2)*0.5, y+h+0.0001, "p < 0.001", ha='center', va='bottom', color='black', fontsize=6, fontproperties=arial_font)

plt.tight_layout()
plt.savefig("Figure_3a_v1.svg", format="svg", dpi=300)
plt.show()

```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# =========================
# Load and register Arial
# =========================
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"

fm.fontManager.addfont(arial_path)
arial_font = fm.FontProperties(fname=arial_path)
arial_name = arial_font.get_name()

plt.rcParams['font.family'] = arial_name
plt.rcParams['font.sans-serif'] = [arial_name]

# Font sizes (same system as other figures)
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


# =========================
# Colors and order
# =========================
color_map = {
    'PJ':'#E4F1F8','MP': '#C5E1EF','KP': '#3A93C2','PM': '#0C4A70',
    'BJ': '#6CB0D6','SC': '#9EC9E2','PK': '#246D9C','VS': '#0A592A',
    'TM': '#3FAD5B','RG': '#218B3B','SP': '#CDE5D2','TR': '#9CCEA7','PL': '#6CB97D'
}

desired_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]


# =========================
# Figure size
# =========================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(12.3*cm, 6*cm), dpi=300)


# =========================
# Grid
# =========================
ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

ax.grid(
    which='minor',
    axis='y',
    linestyle='--',
    linewidth=0.3,
    color='grey',
    alpha=0.5
)


# =========================
# Strip plot
# =========================
sns.stripplot(
    data=df_angsd,
    x='population',
    y='mean_100KB_het',
    palette=color_map,
    jitter=True,
    size=8,
    order=desired_order,
    alpha=0.5,
    zorder=1,
    ax=ax
)

ax.xaxis.set_tick_params(which='minor', bottom=False)


# =========================
# Median lines
# =========================
stats = df_angsd.groupby('population')['mean_100KB_het'].median().reindex(desired_order)

for i, pop in enumerate(desired_order):

    ax.hlines(
        y=stats.loc[pop],
        xmin=i-0.3,
        xmax=i+0.3,
        color=color_map[pop],
        linewidth=3,
        zorder=2
    )


# --- Y axis cleanup ---
#ax.set_ylabel("")
ax.tick_params(
    axis='y',
    which='major',
    length=3,        # shorter ticks
    width=0.8,       # thinner
    direction='out', # cleaner look
    pad=2
)

ax.tick_params(
    axis='x',
    which='major',
    bottom=True,
    length=3,
    width=0.8,
    direction='out',
    pad=2
)

ax.set_xlabel("Population")
ax.set_ylabel("Heterozygosity")

# --- Black spines ---
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(0.8)

# =========================
# Layout (IDENTICAL)
# =========================
ax.set_position([0.10, 0.18, 0.88, 0.74])


# =========================
# Save
# =========================
fig.canvas.draw()

plt.savefig(
    "Figure_S2.png",
    format="png",
    dpi=300
)

plt.show()
```


```python
mergedH = pd.merge(
    df_angsd,
    hEst_df3,
    left_on="individual",
    right_on="Individual_ID",
    how="inner"
)
print(mergedH)
```


```python
plt.figure(figsize=(6,6))


sns.scatterplot(
    data=mergedH,
    x="mean_100KB_het", # het ANGSD
    y="Mean_h" # mean_h ROHan
)


plt.xlabel("het (ANGSD)")
plt.ylabel("Mean_h (ROHan)")
plt.tight_layout()
plt.show()
```


```python
corr = mergedH["mean_100KB_het"].corr(mergedH["Mean_h"]) # very good correlation!
print(f"Correlation between het ANGSD and ROHan mean h: {corr:.3f}")
```


```python
mergedH2 = pd.merge(
    df_angsd,
    rohan_df2,
    left_on="individual",
    right_on="individual",
    how="inner"
)
print(mergedH2)
```


```python
plt.figure(figsize=(6,6))


sns.scatterplot(
    data=mergedH2,
    x="mean_100KB_het", # het ANGSD
    y="segments_roh_percent_mid" # mean_h ROHan
)


plt.xlabel("het (ANGSD)")
plt.ylabel("%ROH (ROHan)")
plt.tight_layout()
plt.show()
```


```python
corr = mergedH2["mean_100KB_het"].corr(mergedH2["segments_in_roh_percent_mid"]) # okay correlation
print(f"Correlation between het ANGSD and ROHan %ROH: {corr:.3f}")
```


```python
mergedH3 = pd.merge(
    df_angsd,
    roh_per_ind,
    left_on="individual",
    right_on="Sample",
    how="inner"
)
print(mergedH3)
```


```python
plt.figure(figsize=(6,6))


sns.scatterplot(
    data=mergedH3,
    x="mean_100KB_het", # het ANGSD
    y="percent_genome_in_roh" # mean_h ROHan
)

plt.xlabel("het (ANGSD)")
plt.ylabel("%ROH (bcftools ROH)")
plt.tight_layout()
plt.show()
```


```python
corr = mergedH3["mean_100KB_het"].corr(mergedH3["percent_genome_in_roh"]) # very good correlation
print(f"Correlation between het ANGSD and bcftools %ROH: {corr:.3f}")
```

## 8. PHLASH

### 8.3. PHLASH PLOTS


```python
color_map = {
    'PJ':'#E4F1F8',
    'MP': '#C5E1EF',
    'KP': '#3A93C2',
    'PM': '#0C4A70',
    'BJ': '#6CB0D6',
    'SC': '#9EC9E2',
    'PK': '#246D9C',
    'VS': '#0A592A',
    'TM': '#3FAD5B',
    'RG': '#218B3B',
    'SP': '#CDE5D2',
    'TR': '#9CCEA7',
    'PL': '#6CB97D',
    'IT': '#218B3B',
    'HR': '#9CCEA7'
}
```


```python
# ==============================
# CONFIGURATION
# ==============================

GEN_TIME = 2  # years per generation

METADATA = [
    {"name": "Island BJ", "output_dir": "results/BJ"},
    {"name": "Island KP", "output_dir": "results/KP"},
    {"name": "Island MP", "output_dir": "results/MP"},
    {"name": "Island PJ", "output_dir": "results/PJ"},
    {"name": "Island PK (all years)", "output_dir": "results/PK_all"},
    {"name": "Island PM (all years)", "output_dir": "results/PM_all"},
    {"name": "Island SC (all years)", "output_dir": "results/SC_all"},
    {"name": "Mainland HR", "output_dir": "results/HR"},
    {"name": "Mainland IT", "output_dir": "results/IT"},
]

BASE_RESULTS_DIR = "../results/demographic_analyses"
ARIAL_PATH = "/scratch/antwerpen/206/vsc20666/arial.ttf"

# =========================
# Style
# =========================
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

fm.fontManager.addfont(ARIAL_PATH)
arial = fm.FontProperties(fname=ARIAL_PATH)
arial_name = arial.get_name()

plt.rcParams.update({
    "font.family": arial_name,
    "font.size": 6,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# =========================
# Colors
# =========================
COLOR_MAP = {
    'PJ':'#E4F1F8',
    'MP': '#C5E1EF',
    'KP': '#3A93C2',
    'PM': '#0C4A70',
    'BJ': '#6CB0D6',
    'SC': '#9EC9E2',
    'PK': '#246D9C',
    'VS': '#0A592A',
    'TM': '#3FAD5B',
    'RG': '#218B3B',
    'SP': '#CDE5D2',
    'TR': '#9CCEA7',
    'PL': '#6CB97D',
    'IT': '#218B3B',
    'HR': '#9CCEA7'
}
DEFAULT_COLOR = "#999999"

# =========================
# Climate events (years)
# =========================
CLIMATE_EVENTS = [
    {"name": "LGM", "start": 19000, "end": 26000},
    {"name": "PGM", "start": 130000, "end": 150000},
    {"name": "MPT", "start": 550000, "end": 1250000},
]

SHOW_EVENT_LABELS = False

# =========================
# Helpers
# =========================
def get_population_code(name):
    if "Island" in name:
        return name.split("Island ")[1].split(" ")[0]
    if "Mainland" in name:
        return name.split("Mainland ")[1].split(" ")[0]
    return name

def get_color(code):
    if code == "IT":
        return COLOR_MAP["VS"]
    if code == "HR":
        return COLOR_MAP["SP"]
    return COLOR_MAP.get(code.replace("_all", ""), DEFAULT_COLOR)

def load_posterior_ne(path):
    data = np.load(path)
    times = data["times"]
    ne = data["ne"]

    median = np.quantile(ne, 0.5, axis=0)
    low = np.quantile(ne, 0.025, axis=0)
    high = np.quantile(ne, 0.975, axis=0)

    return times, median, low, high

# =========================
# Plot
# =========================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(12.3*cm, 8*cm), dpi=300)

# --- Climate bands ---
# --- Climate bands ---
for event in CLIMATE_EVENTS:
    start = event["start"] / GEN_TIME
    end = event["end"] / GEN_TIME

    ax.axvspan(
        start,
        end,
        color="#949494",     # ← updated grey
        alpha=0.3,
        edgecolor='none',    # ← remove border
        linewidth=0,         # ← ensure no edge rendering
        zorder=1
    )

    if SHOW_EVENT_LABELS:
        ax.text(
            (start + end) / 2,
            0.92,
            event["name"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6
        )

# --- Grid ---
ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
ax.set_axisbelow(True)

# --- Plot populations ---
for pop in METADATA:
    path = os.path.join(
        BASE_RESULTS_DIR,
        pop["output_dir"],
        "posterior_ne.npz"
    )

    if not os.path.exists(path):
        continue

    x, y, y_low, y_high = load_posterior_ne(path)

    code = get_population_code(pop["name"])
    color = get_color(code)

    # CI
    ax.plot(x, y_low, color=color, lw=0.6, linestyle="--", alpha=0.8, zorder=2)
    ax.plot(x, y_high, color=color, lw=0.6, linestyle="--", alpha=0.8, zorder=2)

    # Median
    ax.plot(x, y, color=color, lw=1.5, zorder=3)

# =========================
# Axes (FIXED)
# =========================
ax.set_xscale("log")   # ← critical fix
ax.set_yscale("log")

ax.set_xlim(1, 1e7)
ax.set_ylim(1e2, 1e9)

# labels (kept minimal like your clean version)
ax.set_xlabel("")
ax.set_ylabel("")

#ax.tick_params(axis='both', which='major', length=4, width=0.8, direction='out')
#ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')
ax.tick_params(
    axis='both',
    which='both',
    labelbottom=False,
    labelleft=False
)

# =========================
# Spines
# =========================
# =========================
# Spines (black frame)
# =========================
for side in ["top", "bottom", "left", "right"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1)
    ax.spines[side].set_color("black")
for spine in ax.spines.values():
    spine.set_zorder(3)

# =========================
# Layout
# =========================
ax.set_position([0.10, 0.10, 0.85, 0.85])

# =========================
# Save
# =========================
plt.savefig("posterior_ne_with_climate.svg", format="svg", dpi=300)
plt.show()
```
