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


```python
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
import matplotlib.pyplot as plt

# Load sample IDs
with open("../results/population_analyses/rPodSic1.hap1.1/tree/NJ/psiculus_inbreeding.mdist.id") as f:
    sample_ids = [line.strip().replace(" ", "_") for line in f]  # sanitize names

# Load distance matrix (skip row labels if present)
dmatrix = []
with open("../results/population_analyses/rPodSic1.hap1.1/tree/NJ/psiculus_inbreeding.mdist") as f:
    for i, line in enumerate(f):
        row = line.strip().split()
        # Remove sample label if present (first element)
        if row[0] == sample_ids[i]:
            row = row[1:]
        # Keep only the lower triangle part
        dmatrix.append([float(x) for x in row[:i+1]])

# Create Biopython DistanceMatrix
dm = DistanceMatrix(sample_ids, dmatrix)

# Build NJ tree
constructor = DistanceTreeConstructor()
nj_tree = constructor.nj(dm)

# Optional: root with outgroup
if 'PMU01' in sample_ids:
    nj_tree.root_with_outgroup('PMU01')

# Save tree as Newick
Phylo.write(nj_tree, "psiculus_inbreeding_plink_nj_tree.nwk", "newick")

# Plot tree
Phylo.draw(nj_tree)
```


```python
chromosomes = [f'OZ0768{i}.1' for i in range(38,58) if i!= 52 if i!= 56]
chrom_length = pd.read_csv('/scratch/antwerpen/grp/asvardal/projects/psiculus/genome/GCA_964188175.1_rPodSic1.hap1.1_genomic.fna.fai',
                           sep='\t',usecols=[0,1],names=['chrom','len'],index_col=0)
print(chrom_length.iloc[0:20])
```


```python
pwd_path = "/scratch/antwerpen/grp/asvardal/projects/psiculus/results/population_analyses/rPodSic1.hap1.1/tree/NJ/pairwise_differences_matrix_full.csv"
pwd_df = pd.read_csv(pwd_path, index_col=0)

# Confirm it's square and symmetric
assert pwd_df.shape[0] == pwd_df.shape[1]
assert (pwd_df.values == pwd_df.values.T).all()

samples = pwd_df.index.tolist()
pwd_matrix = pwd_df.values
```


```python
tree = diversity.get_nj_tree(
    pairwise_differences=pwd_matrix,
    samples=samples,
    outgroup="PMU01",
    prune_outgroup=False  
)
```


```python
newick_str = tree.write(format=1)[0]
with open("psiculus_inbreeding_nj_tree.nwk", "w") as f:
    f.write(newick_str)
```


```python
tree_path = "psiculus_inbreeding_nj_tree.nwk"
tree = Phylo.read(tree_path, "newick")
fig = plt.figure(figsize=(15, 30))  # adjust size based on number of taxa
axes = fig.add_subplot(1, 1, 1)
Phylo.draw(tree, do_show=False, axes=axes)
plt.show()

```


```python
tree.set_outgroup("PMU01", end_at_present=False)
pwd_window_df = pd.read_csv("/scratch/antwerpen/grp/asvardal/projects/psiculus/results/population_analyses/rPodSic1.hap1.1/tree/NJ/pairwise_differences_matrix_windows.csv", index_col=[0, 1])

```


```python
support_tree, support_series = diversity.get_bootstrap_support(
    real_tree=tree,
    pwd_window=pwd_window_df,
    n_bootstrap_samples=100,
    outgroup="PMU01"  # or whichever sample you want to root on
)
```


```python
for node in support_tree.iter_descendants():
    if not node.is_leaf():
        print(f"{node.get_name()}: {getattr(node, 'pct_support', None)}% support")
```


```python
newick_str = support_tree.write(format=1)[0]  # Extract from tuple if needed
output_path = "/scratch/antwerpen/grp/asvardal/projects/psiculus/notebooks/psiculus_inbreeding_nj_tree_bootstrap.nwk"

with open(output_path, "w") as f:
    f.write(newick_str)

print(f"✅ Bootstrapped Newick tree saved to: {output_path}")

```


```python
# Assign bootstrap support to node names
for node in support_tree.iter_descendants():
    if not node.is_leaf():
        node.name = str(int(getattr(node, "pct_support", 0)))

# Get Newick string (unpack from tuple)
newick_str = support_tree.write(format=1)[0]

# Write to file
output_path = "/scratch/antwerpen/grp/asvardal/projects/psiculus/notebooks/psiculus_inbreeding_nj_tree_bootstrap.nwk"
with open(output_path, "w") as f:
    f.write(newick_str)

print(f"✅ Bootstrapped Newick tree saved to: {output_path}")

```


```python

```


```python
from Bio import Phylo
import matplotlib.pyplot as plt

# =========================
# Load tree
# =========================
tree = Phylo.read("tree.tre", "newick")

# =========================
# Root with outgroup
# =========================
tree.root_with_outgroup({"name": "PMU"})

# =========================
# Remove outgroup
# =========================
tree.prune("PMU")

# =========================
# Remove bootstrap/support labels
# =========================
for clade in tree.find_clades():
    clade.confidence = None

# =========================
# Figure size
# =========================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(9.53*cm, 2.55*cm))

# =========================
# Draw tree
# =========================
Phylo.draw(
    tree,
    axes=ax,
    do_show=False,
    label_func=lambda x: None
)

# round branch corners
for line in ax.get_lines():
    line.set_solid_joinstyle("round")
    line.set_solid_capstyle("round")

# remove axes
ax.set_axis_off()

# fill figure
ax.set_position([0, 0, 1, 1])
ax.margins(0)

plt.savefig("tree_branches_only.svg", bbox_inches='tight', pad_inches=0)
plt.show()
```


```python

```


```python

```


```python

```


```python

```


```python

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


```python
# set your folder path
rohan_path = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/rohan/'
# list of files
files = [f for f in os.listdir(rohan_path) if f.endswith("summary.txt")]
```


```python
# patterns for theta lines with three values
theta_patterns = {
    "theta_outside_ROH": r"Genome-wide theta outside ROH:\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
    "theta_inside_ROH": r"Genome-wide theta inc\. ROH:\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
}

# patterns for other lines: (main value regex, CI regex)
metric_patterns = {
    "segments_unclassified": r"Segments unclassified\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_unclassified_percent": r"Segments unclassified \(%\):\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_in_roh": r"Segments in ROH\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_in_roh_percent": r"Segments in ROH\(%\)\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_non_roh": r"Segments in non-ROH\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_non_roh_percent": r"Segments in non-ROH \(%\)\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "avg_roh_length": r"Avg\. length of ROH\s*:\s*([\d.eE+-]+)\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)"
}

rohan_data = []

for file in files:
    file_path = os.path.join(rohan_path, file)
    with open(file_path, 'r') as f:
        text = f.read()

    data = {"individual": file.replace(".summary.txt", "")}

    # Handle theta values with three fields
    for key, pattern in theta_patterns.items():
        match = re.search(pattern, text)
        if match:
            data[f"{key}_mid"] = float(match.group(1))
            data[f"{key}_min"] = float(match.group(2))
            data[f"{key}_max"] = float(match.group(3))
        else:
            data[f"{key}_mid"] = None
            data[f"{key}_min"] = None
            data[f"{key}_max"] = None

    # Handle other fields
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, text)
        if match:
            data[f"{key}_mid"] = float(match.group(1))
            data[f"{key}_min"] = float(match.group(2))
            data[f"{key}_max"] = float(match.group(3))
        else:
            data[f"{key}_mid"] = data[f"{key}_min"] = data[f"{key}_max"] = None

    rohan_data.append(data)

# Create DataFrame and export
rohan_df = pd.DataFrame(rohan_data)
#rohan_df.to_excel("ROHan_summary.xlsx", index=False)
rohan_df.head()
```


```python
# Extract population code from Individual_ID
rohan_df['population'] = rohan_df['individual'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))

# Step 1: collapse "24XX" into "XX"
rohan_df['population'] = rohan_df['population'].str.replace(r'^24', '', regex=True)

# Step 2: merge CV and FV into RG
rohan_df['population'] = rohan_df['population'].replace({'CV': 'RG', 'FV': 'RG'})
rohan_df.head()

# Step 3: remove BJ18 and RG08
#rohan_df = rohan_df[~rohan_df['individual'].isin(['BJ18', 'RG08'])]
rohan_df.head()
```


```python
# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"  # adjust if this is where your Arial is
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

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

# === Make the plot ===
cm = 1 / 2.54  # cm to inches
plt.figure(figsize=(15*cm, 10*cm))  


# Strip plot first (dots in the background)
import matplotlib.ticker as mticker

plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)  # major grid
plt.minorticks_on()

plt.grid(
    which='minor',
    axis='y', # minor grid lines
    linestyle='--', 
    linewidth=0.3, 
    color='grey', 
    alpha=0.5
)
ax = sns.stripplot(
    data=rohan_df,
    x='population',
    y='segments_in_roh_percent_mid',
    palette=color_map,
    jitter=True,
    size=6,
    order=desired_order,
    alpha=0.3,  # make dots slightly transparent
    zorder=1     # put dots behind lines
)
ax.xaxis.set_tick_params(which='minor', bottom=False)

# Calculate median per population
stats = rohan_df.groupby('population')['segments_in_roh_percent_mid'].median().reindex(desired_order)

# Add median lines with same color as population
for i, pop in enumerate(desired_order):
    median = stats.loc[pop]
    color = color_map[pop]  # get color for this population
    
    ax.hlines(
        y=median,
        xmin=i - 0.3,
        xmax=i + 0.3,
        color=color,
        linewidth=3,
        zorder=2  # above dots
    )

ax.set_ylabel("% genome in ROHs", fontsize = 8, fontproperties=arial_font)
ax.tick_params(axis='x', rotation=45)
ax.set_xlabel("")
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12], labels=["", "","","","","","","","","","","",""], rotation=0, fontsize = 6)
plt.yticks(fontsize = 6)

plt.tight_layout()
plt.savefig("Figure_3b_v1.svg", format="svg", dpi=300)
plt.show()

```


```python
# First, create Population column, extracting letters from Individual_ID
hEst_df['Population'] = hEst_df['Individual_ID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
# Reassign CV and FV to RG
hEst_df.loc[hEst_df['Population'].isin(['CV', 'FV']), 'Population'] = 'RG'
# Define custom population order
pop_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]
```


```python
# Plot boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=hEst_df, x='Population', y='Mean_h', order=pop_order)
plt.title('Heterozygosity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


```python
### See also R

# Extract population code from Individual_ID
hEst_df['Population'] = hEst_df['Individual_ID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))

# Step 1: collapse "24XX" into "XX"
hEst_df['Population'] = hEst_df['Population'].str.replace(r'^24', '', regex=True)

# Step 2: merge CV and FV into RG
hEst_df['Population'] = hEst_df['Population'].replace({'CV': 'RG', 'FV': 'RG'})
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

hEst_df['Region'] = hEst_df['Population'].apply(assign_region)
hEst_df.to_excel("hEst_summary.xlsx", index=False)
```


```python
import statsmodels.formula.api as smf
# Fit a mixed-effects model: Region as fixed effect, Population as random effect
model = smf.mixedlm("Mean_h ~ Region", hEst_df, groups=hEst_df["Population"])
result = model.fit()

print(result.summary())
```


```python
model = smf.ols("Mean_h ~ Region", data=hEst_df).fit()
print(model.summary())
```


```python
# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"  # adjust if this is where your Arial is
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

# === Define colors ===
palette = {"Mainland": "#009900", "Island": "#0066CC"}

# === Make the plot ===
cm = 1 / 2.54  # cm to inches
plt.figure(figsize=(10*cm, 10*cm))  

ax = sns.boxplot(
    data=hEst_df,
    x="Region",
    y="Mean_h",
    hue="Region",
    palette=palette,
    linewidth=1,       # box edge width
    fliersize=3,         # outlier size
    flierprops={"marker": "o", "markerfacecolor": "black", "markersize": 4},
    medianprops={"color": "black", "linewidth": 2},  # median line
    #boxprops={"facecolor":"none", "edgecolor":"black"}  # make boxes transparent or change edge
)

# Add dashed grey grid
plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

plt.ylabel("Heterozygosity", fontsize = 8, fontproperties=arial_font)
plt.xlabel("")
plt.xticks(ticks=[0,1], labels=["", ""], rotation=0, fontsize = 6)
plt.yticks(fontsize = 6)
# --- Add statistical annotation ---
p_value = 0.0  # MixedLM p-value for Region effect is <0.0001, show as 0.0 or "<0.001"
x1, x2 = 0, 1  # positions of the boxes

# Compute max y value (top of plot) and offset
y_max = hEst_df['Mean_h'].max()
y = y_max + 0.0005  # start just above max value
h = 0.00005           # height of vertical ticks
line_width = 0.5     # shorter side lines

# Draw short side lines
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')

# Add p-value text slightly above the line
ax.text((x1+x2)*0.5, y+h+0.0001, "p < 0.001", ha='center', va='bottom', color='black', fontsize=6, fontproperties=arial_font)


plt.tight_layout()
#plt.savefig("Figure_3a_v1.svg", format="svg", dpi=300)
plt.show()
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
plt.figure(figsize=(10*cm, 10*cm))

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
    data=hEst_df,
    x='Region',
    y='Mean_h',
    hue='Region',
    palette=palette,
    jitter=True,
    size=6,
    alpha=0.3,  # slightly transparent dots
    zorder=1,    # behind median lines
    dodge=False
)
ax.xaxis.set_tick_params(which='minor', bottom=False)  # remove minor x ticks

# --- Add median lines per Region ---
stats = hEst_df.groupby('Region')['Mean_h'].median().reindex(palette.keys())

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
y_max = hEst_df['Mean_h'].max()
y = y_max + 0.0005  # start just above max value
h = 0.00005          # height of short vertical ticks
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c='black')
ax.text((x1+x2)*0.5, y+h+0.0001, "p < 0.001", ha='center', va='bottom', color='black', fontsize=6, fontproperties=arial_font)

plt.tight_layout()
plt.savefig("Figure_3a_v1.svg", format="svg", dpi=300)
plt.show()

```


```python
merged = pd.merge(
    rohan_df,
    hEst_df,
    left_on="individual",
    right_on="Individual_ID",
)
print(merged)
#merged = merged[~merged['individual'].isin(['BJ18', 'RG08', 'BJ04', 'BJ06', 'BJ09', 'BJ16'])]
```


```python
plt.figure(figsize=(6,6))
sns.scatterplot(
    data=merged,
    x="segments_in_roh_percent_mid", # %genome in ROH, ! outliers removed
    y="Mean_h" # h
)
plt.xlabel("% genome in ROH (ROHan)")
plt.ylabel("Mean h")
plt.tight_layout()
plt.show()
```


```python
corr = merged["Mean_h"].corr(merged["segments_in_roh_percent_mid"]) # outliers removed
print(f"Correlation between Mean h and ROHan %ROH: {corr:.3f}")
```


```python
# set your folder path
rohan_path2 = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/rohan_v2/'
# list of files
files2 = [f for f in os.listdir(rohan_path2) if f.endswith("summary.txt")]
```


```python
# patterns for theta lines with three values
theta_patterns = {
    "theta_outside_ROH": r"Genome-wide theta outside ROH:\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
    "theta_inside_ROH": r"Genome-wide theta inc\. ROH:\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
}

# patterns for other lines: (main value regex, CI regex)
metric_patterns = {
    "segments_unclassified": r"Segments unclassified\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_unclassified_percent": r"Segments unclassified \(%\):\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_in_roh": r"Segments in ROH\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_in_roh_percent": r"Segments in ROH\(%\)\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_non_roh": r"Segments in non-ROH\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "segments_non_roh_percent": r"Segments in non-ROH \(%\)\s*:\s*([\d.eE+-]+)\s*\(([\d.eE+-]+),\s*([\d.eE+-]+)\)",
    "avg_roh_length": r"Avg\. length of ROH\s*:\s*([\d.eE+-]+)\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)"
}

rohan_data = []

for file in files2:
    file_path = os.path.join(rohan_path2, file)
    with open(file_path, 'r') as f:
        text = f.read()

    data = {"individual": file.replace(".summary.txt", "")}

    # Handle theta values with three fields
    for key, pattern in theta_patterns.items():
        match = re.search(pattern, text)
        if match:
            data[f"{key}_mid"] = float(match.group(1))
            data[f"{key}_min"] = float(match.group(2))
            data[f"{key}_max"] = float(match.group(3))
        else:
            data[f"{key}_mid"] = None
            data[f"{key}_min"] = None
            data[f"{key}_max"] = None

    # Handle other fields
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, text)
        if match:
            data[f"{key}_mid"] = float(match.group(1))
            data[f"{key}_min"] = float(match.group(2))
            data[f"{key}_max"] = float(match.group(3))
        else:
            data[f"{key}_mid"] = data[f"{key}_min"] = data[f"{key}_max"] = None

    rohan_data.append(data)

# Create DataFrame and export
rohan_df2 = pd.DataFrame(rohan_data)
rohan_df2.to_excel("ROHan_summary_v2.xlsx", index=False)
rohan_df2.head()
```


```python
# Extract population code from Individual_ID
rohan_df2['population'] = rohan_df2['individual'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))

# Step 1: collapse "24XX" into "XX"
rohan_df2['population'] = rohan_df2['population'].str.replace(r'^24', '', regex=True)

# Step 2: merge CV and FV into RG
rohan_df2['population'] = rohan_df2['population'].replace({'CV': 'RG', 'FV': 'RG'})
rohan_df2.head()

# Step 3: remove BJ18 and RG08
rohan_df2 = rohan_df2[~rohan_df2['individual'].isin(['PMU01'])]
rohan_df2.head()
```


```python
# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"  # adjust if this is where your Arial is
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

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

# === Make the plot ===
cm = 1 / 2.54  # cm to inches
plt.figure(figsize=(15*cm, 10*cm))  


# Strip plot first (dots in the background)
import matplotlib.ticker as mticker

plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)  # major grid
plt.minorticks_on()

plt.grid(
    which='minor',
    axis='y', # minor grid lines
    linestyle='--', 
    linewidth=0.3, 
    color='grey', 
    alpha=0.5
)
ax = sns.stripplot(
    data=rohan_df2,
    x='population',
    y='segments_in_roh_percent_mid',
    palette=color_map,
    jitter=True,
    size=6,
    order=desired_order,
    alpha=0.3,  # make dots slightly transparent
    zorder=1     # put dots behind lines
)
ax.xaxis.set_tick_params(which='minor', bottom=False)

# Calculate median per population
stats = rohan_df2.groupby('population')['segments_in_roh_percent_mid'].median().reindex(desired_order)

# Add median lines with same color as population
for i, pop in enumerate(desired_order):
    median = stats.loc[pop]
    color = color_map[pop]  # get color for this population
    
    ax.hlines(
        y=median,
        xmin=i - 0.3,
        xmax=i + 0.3,
        color=color,
        linewidth=3,
        zorder=2  # above dots
    )

ax.set_ylabel("% genome in ROHs", fontsize = 8, fontproperties=arial_font)
ax.tick_params(axis='x', rotation=45)
ax.set_xlabel("")
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12], labels=["", "","","","","","","","","","","",""], rotation=0, fontsize = 6)
plt.yticks(fontsize = 6)

plt.tight_layout()
#plt.savefig("Figure_3b_v1.svg", format="svg", dpi=300)
plt.show()

```


```python
# folder path
input_directory = '../results/population_analyses/rPodSic1.hap1.1/genetic_diversity/rohan_v2/'
```


```python
# create a list to store all individuals' data
hEst_data = []

# loop through all .hEst.gz files in the specified directory
for input_file in os.listdir(input_directory):
    if input_file.endswith(".hEst.gz"):
        # Full path to the input file
        input_file_path = os.path.join(input_directory, input_file)
        
        # Extract the base ID (e.g., "VS12" from "VS12.hEst.gz")
        base_id = input_file.split('.')[0]
        
        # open with pysam (works for BGZF)
        with pysam.BGZFile(input_file_path, 'r') as f:
            # decode bytes to text, then read with pandas
            df = pd.read_csv(f, sep="\t", comment='#', header=None)
        
        # Extract the 5th column (index 4) for 'h' values
        h_values = df.iloc[:, 4].dropna()  # Access the 5th column directly by index
        
        # Calculate total mean and standard error (SE)
        total_mean_h = np.mean(h_values)
        total_se_h = np.std(h_values, ddof=1) / np.sqrt(len(h_values))

        # Append the results to the list
        hEst_data.append([base_id, total_mean_h, total_se_h])

# Create a DataFrame with the summary data
hEst_df2 = pd.DataFrame(hEst_data, columns=["Individual_ID", "Mean_h", "SE_h"])

# Save the DataFrame to an Excel file
#hEst_df2.to_excel("hEst_summary2.xlsx", index=False)

# Display the first few rows of the summary
hEst_df2.head()
    # outliers in ROH do not form problem here!
```


```python
import os
import pandas as pd
import numpy as np
import pysam

# List of individuals to skip
skip_individuals = ["PMU01"]

# Create a list to store all individuals' data
hEst_data = []

# Loop through all .hEst.gz files in the specified directory
for input_file in os.listdir(input_directory):
    if input_file.endswith(".hEst.gz"):
        base_id = input_file.split('.')[0]
        
        # Skip if in the skip list
        if base_id in skip_individuals:
            print(f"Skipping {base_id} because outgroup")
            continue
        
        input_file_path = os.path.join(input_directory, input_file)
        
        try:
            # Open with pysam (works for BGZF)
            with pysam.BGZFile(input_file_path, 'r') as f:
                df = pd.read_csv(f, sep="\t", comment='#', header=None)
            
            # Extract the 5th column (index 4) for 'h' values
            h_values = df.iloc[:, 4].dropna()
            
            # Calculate mean and standard error
            total_mean_h = np.mean(h_values)
            total_se_h = np.std(h_values, ddof=1) / np.sqrt(len(h_values))
            
            # Calculate % of times h < 0.003
            percent_below_0_003 = 100 * np.sum(h_values < 0.003) / len(h_values)
            
            # Append the results to the list
            hEst_data.append([base_id, total_mean_h, total_se_h, percent_below_0_003])
        
        except Exception as e:
            print(f"Error reading {base_id}: {e}")
            continue

# Create a DataFrame with the summary data
hEst_df3 = pd.DataFrame(hEst_data, columns=["Individual_ID", "Mean_h", "SE_h", "%_h_below_0.003"])

# Display the first few rows
hEst_df3.head()

```


```python
merged2 = pd.merge(
    rohan_df2,
    hEst_df3,
    left_on="individual",
    right_on="Individual_ID",
    how="inner"
)
print(merged2)
```


```python
plt.figure(figsize=(6,6))


sns.scatterplot(
    data=merged2,
    x="segments_in_roh_percent_mid", # %genome in ROH, ! outliers removed
    y="%_h_below_0.003" # h
)


plt.xlabel("% genome in ROH (bcftools)")
plt.ylabel("%_h_below_0.003 (ROHan)")
plt.tight_layout()
plt.show()
```


```python
corr = merged2["segments_in_roh_percent_mid"].corr(merged2["%_h_below_0.003"]) # okay correlation
print(f"Correlation between ROHan %ROH and ROHan h value below 0.003: {corr:.3f}")
```


```python

```


```python
plt.figure(figsize=(6,6))


sns.scatterplot(
    data=merged2,
    x="segments_in_roh_percent_mid", # %genome in ROH, ! outliers removed
    y="Mean_h" # h
)


plt.xlabel("% genome in ROH (ROHan)")
plt.ylabel("Mean_h (ROHan)")
plt.tight_layout()
plt.show()
```


```python
corr = merged2["segments_in_roh_percent_mid"].corr(merged2["Mean_h"]) # okay correlation
print(f"Correlation between ROHan %ROH and ROHan h value below 0.003: {corr:.3f}")
```


```python

```


```python
merged = pd.merge(
    roh_per_ind,
    hEst_df3,
    left_on="Sample",
    right_on="Individual_ID",
    how="inner"
)
print(merged)
```


```python
plt.figure(figsize=(6,6))

# Add a regression line
sns.regplot(
    data=merged,
    x="percent_genome_in_roh",
    y="%_h_below_0.003",
    scatter=False,   # don't plot points again
    color='red'
)
sns.scatterplot(
    data=merged,
    x="percent_genome_in_roh", # %genome in ROH, ! outliers removed
    y="%_h_below_0.003" # h
)

# Add y=x line
max_val = max(merged["percent_genome_in_roh"].max(), merged["%_h_below_0.003"].max())
plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--')

plt.xlabel("% genome in ROH (bcftools)")
plt.ylabel("%_h_below_0.003 (ROHan)")
plt.tight_layout()
plt.show()
```


```python
corr = merged["percent_genome_in_roh"].corr(merged["%_h_below_0.003"]) # very good!
print(f"Correlation between bcftools and ROHan %_h_below_0.003: {corr:.3f}")
```


```python
corr = merged["percent_genome_in_roh"].corr(merged["Mean_h"]) # very good!
print(f"Correlation between bcftools and ROHan Mean_h: {corr:.3f}")
```


```python
### See also R
# Extract population code from Individual_ID
hEst_df2['Population'] = hEst_df2['Individual_ID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))

# Step 1: collapse "24XX" into "XX"
hEst_df2['Population'] = hEst_df2['Population'].str.replace(r'^24', '', regex=True)

# Step 2: merge CV and FV into RG
hEst_df2['Population'] = hEst_df2['Population'].replace({'CV': 'RG', 'FV': 'RG'})
hEst_df2 = hEst_df2[~hEst_df2['Individual_ID'].isin(['PMU01'])]
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

hEst_df2['Region'] = hEst_df2['Population'].apply(assign_region)
hEst_df2.to_excel("hEst_summary2.xlsx", index=False)
```


```python
# === Load Arial font ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"  # adjust if this is where your Arial is
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()

# === Define colors ===
palette = {"Mainland": "#009900", "Island": "#0066CC"}

# === Make the plot ===
cm = 1 / 2.54  # cm to inches
plt.figure(figsize=(10*cm, 10*cm))  

ax = sns.boxplot(
    data=hEst_df2,
    x="Region",
    y="Mean_h",
    hue="Region",
    #palette=palette,
    linewidth=1,       # box edge width
    fliersize=3,         # outlier size
    flierprops={"marker": "o", "markerfacecolor": "black", "markersize": 4},
    medianprops={"color": "black", "linewidth": 2},  # median line
    #boxprops={"facecolor":"none", "edgecolor":"black"}  # make boxes transparent or change edge
)

# Add dashed grey grid
plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

plt.ylabel("Heterozygosity", fontsize = 8, fontproperties=arial_font)
plt.xlabel("")
plt.xticks(ticks=[0,1], labels=["", ""], rotation=0, fontsize = 6)
plt.yticks(fontsize = 6)
# --- Add statistical annotation ---
p_value = 0.0  # MixedLM p-value for Region effect is <0.0001, show as 0.0 or "<0.001"
x1, x2 = 0, 1  # positions of the boxes

# Compute max y value (top of plot) and offset
y_max = hEst_df2['Mean_h'].max()
y = y_max + 0.0005  # start just above max value
h = 0.00005           # height of vertical ticks
line_width = 0.5     # shorter side lines

# Draw short side lines
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')

# Add p-value text slightly above the line
ax.text((x1+x2)*0.5, y+h+0.0001, "p < 0.001", ha='center', va='bottom', color='black', fontsize=6, fontproperties=arial_font)


plt.tight_layout()
#plt.savefig("Figure_3a_v1.svg", format="svg", dpi=300)
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


```python

```


```python
merged = pd.merge(
    roh_per_ind,
    rohan_df,
    left_on="Sample",
    right_on="individual",
    how="inner"
)
merged = merged[~merged['Sample'].isin(['BJ18', 'RG08', 'BJ04', 'BJ06', 'BJ09', 'BJ16'])]
```


```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.scatterplot(
    data=merged,
    x="percent_genome_in_roh",               # bcftools
    y="segments_in_roh_percent_mid"          # rohan
)
plt.xlabel("% genome in ROH (bcftools)")
plt.ylabel("% genome in ROH (ROHan)")
plt.title("Comparison of %ROH between bcftools and ROHan")
plt.tight_layout()
plt.show()

```


```python
corr = merged["percent_genome_in_roh"].corr(merged["segments_in_roh_percent_mid"]) # with outliers very bad correlation, but without very good!
print(f"Correlation between bcftools and ROHan %ROH: {corr:.3f}")
```


```python
merged2 = pd.merge(
    roh_per_ind,
    hEst_df,
    left_on="Sample",
    right_on="Individual_ID",
)
print(merged2)
#merged2 = merged2[~merged2['Sample'].isin(['BJ18', 'RG08', 'BJ04', 'BJ06', 'BJ09', 'BJ16'])]
```


```python
plt.figure(figsize=(6,6))
sns.scatterplot(
    data=merged2,
    x="percent_genome_in_roh", # %genome in ROH, ! outliers removed
    y="Mean_h" # h
)
plt.xlabel("% genome in ROH (bcftools)")
plt.ylabel("Mean h")
plt.tight_layout()
plt.show()
```


```python
corr = merged2["percent_genome_in_roh"].corr(merged2["Mean_h"]) # with or without outliers very good correlation!
print(f"Correlation between bcftools and ROHan h: {corr:.3f}")
```


```python
merged = pd.merge(
    roh_per_ind,
    rohan_df2,
    left_on="Sample",
    right_on="individual",
    how="inner"
)
print(merged)
#merged = merged[~merged['Sample'].isin(['BJ18', 'RG08', 'BJ04', 'BJ06', 'BJ09', 'BJ16'])]
#merged = merged[merged['segments_in_roh_percent_mid'] != 100]
```


```python

plt.figure(figsize=(6,6))
sns.scatterplot(
    data=merged,
    x="percent_genome_in_roh",               # bcftools
    y="segments_in_roh_percent_mid"          # rohan
)

# Annotate each point with the Sample name
for i, row in merged.iterrows():
    plt.text(
        row['percent_genome_in_roh'],
        row['segments_in_roh_percent_mid'],
        row['Sample'],
        fontsize=8,               # smaller font to avoid clutter
        ha='right',               # horizontal alignment
        va='bottom'               # vertical alignment
    )

plt.xlabel("% genome in ROH (bcftools)")
plt.ylabel("% genome in ROH (ROHan)")
plt.title("Comparison of %ROH between bcftools and ROHan v2")
plt.tight_layout()
plt.show()
```


```python
corr = merged["percent_genome_in_roh"].corr(merged["segments_in_roh_percent_mid"]) 
print(f"Correlation between bcftools and ROHan v2 %ROH: {corr:.3f}")
```


```python
merged2 = pd.merge(
    roh_per_ind,
    hEst_df3,
    left_on="Sample",
    right_on="Individual_ID",
    how="inner"
)
print(merged2)
```


```python
plt.figure(figsize=(6,6))


sns.scatterplot(
    data=merged2,
    x="Mean_h", # het ANGSD
    y="percent_genome_in_roh" # mean_h ROHan
)

plt.xlabel("h (ROHan)")
plt.ylabel("%ROH (bcftools ROH)")
plt.tight_layout()
plt.show()
```


```python
corr = merged2["Mean_h"].corr(merged2["percent_genome_in_roh"]) # good correlation
print(f"Correlation between het rohan h and bcftools %ROH: {corr:.3f}")
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

```


```python

```


```python

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


```python
metadata = [
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

```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter

# --- Enhanced Style Configuration ---
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Create figure with single plot
fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)

# Use tab10 palette - excellent for 9 distinguishable colors
colors = sns.color_palette("tab10", n_colors=9)

# Combine all files
all_files = metadata

max_time = 0
missing_files = []

def plot_population_data(ax, files, colors):
    """Helper function to plot data on a given axis"""
    global max_time
    
    for idx, file in enumerate(files):
        path = f"..//results/demographic_analyses/{file['output_dir']}/posterior_ne.npz"
        
        if not os.path.exists(path):
            missing_files.append(file['name'])
            continue
        
        try:
            data = np.load(path)
            x = data["times"] * 2
            y = np.quantile(data["ne"], 0.5, axis=0)
            y_low = np.quantile(data["ne"], 0.025, axis=0)
            y_high = np.quantile(data["ne"], 0.975, axis=0)
            
            max_time = max(max_time, x[-1])
            
            # Plot with distinct colors and improved styling
            color = colors[idx % len(colors)]
            ax.plot(x, y, lw=2.5, label=file["name"], color=color, zorder=3)
            ax.fill_between(x, y_low, y_high, alpha=0.25, color=color, zorder=2)
            
        except Exception as e:
            print(f"Error loading {file['name']}: {e}")
            missing_files.append(file['name'])

# Plot all populations
plot_population_data(ax, all_files, colors)

# Formatting
ax.set_yscale("log")
ax.set_xscale("symlog")
ax.set_ylabel("Effective population size ($N_e$)", fontsize=13, fontweight='medium')
ax.set_xlabel("Time (years ago)", fontsize=13, fontweight='medium')
ax.set_title("Posterior Effective Population Size Through Time", 
             fontsize=14, fontweight='bold', pad=10)

# Legend below the plot
ax.legend(frameon=True, fontsize=9, loc='upper center', framealpha=0.95,
          edgecolor='gray', fancybox=True, ncol=3, bbox_to_anchor=(0.5, -0.15))

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xlim(10, max_time)

# Tighter layout with extra space for legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# --- Save in multiple formats ---
plt.savefig("posterior_ne_all_islands.pdf", format="pdf", 
            bbox_inches="tight", dpi=300)
plt.savefig("posterior_ne_all_islands.png", format="png", 
            bbox_inches="tight", dpi=300)

# Report missing files if any
if missing_files:
    print(f"\nWarning: {len(missing_files)} file(s) not found:")
    for name in missing_files:
        print(f"  - {name}")

plt.show()
```


```python
# --- Metadata ---
metadata = [
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

# --- Custom color scheme ---
color_map = {
    'PJ': '#6BAED6',
    'MP': '#2171B5',
    'KP': '#4292C6',
    'PM': '#08306B',
    'BJ': '#08519C',
    'SC': '#9ECAE1',
    'PK': '#C6DBEF',
    'VS': '#238B45',  # IT will use this
    'TM': '#00441B',
    'RG': '#BAE4B3',
    'SP': '#74C476',  # HR will use this
    'TR': '#006D2C',
    'PL': '#41AB5D'
}

# === Load Arial font from scratch ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()
```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter


# --- Plot styling ---
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 3
plt.rcParams['axes.titlesize'] = 3

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)

max_time = 0
missing_files = []

def get_population_code(name):
    """Extracts population code from metadata name"""
    if "Island" in name:
        code = name.split("Island ")[1].split(" ")[0]
    elif "Mainland" in name:
        code = name.split("Mainland ")[1].split(" ")[0]
    else:
        code = name
    return code

def get_color(code):
    """Assign custom color for each population"""
    if code == "IT":
        return color_map["VS"]  # IT uses VS color
    elif code == "HR":
        return color_map["SP"]  # HR uses SP color
    else:
        # Handle "(all years)" variants like PK_all → PK
        base_code = code.replace("_all", "")
        return color_map.get(base_code, "#999999")  # default gray fallback

def plot_population_data(ax, files):
    """Helper function to plot data on a given axis"""
    global max_time
    # Grey background
    #ax.set_facecolor("#E5E4E2")
    # White dashed grid behind everything
    #ax.grid(True, color='white', linestyle='--', linewidth=0.7, alpha=1)
    #ax.set_axisbelow(True)
    
    for file in files:
        path = f"..//results/demographic_analyses/{file['output_dir']}/posterior_ne.npz"
        
        if not os.path.exists(path):
            missing_files.append(file["name"])
            continue
        
        try:
            data = np.load(path)
            x = data["times"] * 2
            y = np.quantile(data["ne"], 0.5, axis=0)
            #y_low = np.quantile(data["ne"], 0.025, axis=0)
            #y_high = np.quantile(data["ne"], 0.975, axis=0)
            
            max_time = max(max_time, x[-1])
            
            code = get_population_code(file["name"])
            color = get_color(code)
            
            ax.plot(x, y, lw=2.5, label=file["name"], color=color, zorder=3)
            #ax.fill_between(x, y_low, y_high, alpha=0.25, color=color, zorder=2)
            
        except Exception as e:
            print(f"Error loading {file['name']}: {e}")
            missing_files.append(file['name'])


plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.5)
# --- Plot all populations ---
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
plot_population_data(ax, metadata)


# --- Formatting ---
ax.set_yscale("log")
ax.set_xscale("symlog")
ax.set_ylabel("Effective population size ($N_e$)", fontsize=8, fontweight='medium', fontproperties=arial_font)
ax.set_xlabel("Time (years ago)", fontsize=8, fontweight='medium', fontproperties=arial_font)

# Legend below the plot
#ax.legend(frameon=True, fontsize=9, loc='upper center', framealpha=0.95,
#          edgecolor='gray', fancybox=True, ncol=3, bbox_to_anchor=(0.5, -0.15))


ax.set_axisbelow(True)
ax.set_xlim(10, max_time)


# Layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
#plt.savefig("posterior_ne_all_islands.pdf", format="pdf", bbox_inches="tight", dpi=300)
#plt.savefig("posterior_ne_all_islands.svg", format="svg", bbox_inches="tight", dpi=300)

# Missing file report
if missing_files:
    print(f"\nWarning: {len(missing_files)} file(s) not found:")
    for name in missing_files:
        print(f"  - {name}")

plt.show()

```


```python
# --- Metadata ---
metadata = [
    {"name": "Island PK (all years)", "output_dir": "results/PK_all"},
    {"name": "Island PM (all years)", "output_dir": "results/PM_all"},
]

# --- Custom color scheme ---
color_map = {
    'PJ': '#6BAED6',
    'MP': '#2171B5',
    'KP': '#4292C6',
    'PM': '#08306B',
    'BJ': '#08519C',
    'SC': '#9ECAE1',
    'PK': '#C6DBEF',
    'VS': '#238B45',  # IT will use this
    'TM': '#00441B',
    'RG': '#BAE4B3',
    'SP': '#74C476',  # HR will use this
    'TR': '#006D2C',
    'PL': '#41AB5D'
}

# === Load Arial font from scratch ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()
```


```python
# --- Metadata ---
metadata = [
    {"name": "Island MP", "output_dir": "results/MP"},
    {"name": "Mainland IT", "output_dir": "results/IT"}]

# --- Custom color scheme ---
color_map = {
    'PJ': '#6BAED6',
    'MP': '#2171B5',
    'KP': '#4292C6',
    'PM': '#08306B',
    'BJ': '#08519C',
    'SC': '#9ECAE1',
    'PK': '#C6DBEF',
    'VS': '#238B45',  # IT will use this
    'TM': '#00441B',
    'RG': '#BAE4B3',
    'SP': '#74C476',  # HR will use this
    'TR': '#006D2C',
    'PL': '#41AB5D'
}

# === Load Arial font from scratch ===
arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
arial_font = fm.FontProperties(fname=arial_path)
plt.rcParams['font.family'] = arial_font.get_name()
```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter


# --- Plot styling ---
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 6

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)

max_time = 0
missing_files = []

def get_population_code(name):
    """Extracts population code from metadata name"""
    if "Island" in name:
        code = name.split("Island ")[1].split(" ")[0]
    elif "Mainland" in name:
        code = name.split("Mainland ")[1].split(" ")[0]
    else:
        code = name
    return code

def get_color(code):
    """Assign custom color for each population"""
    if code == "IT":
        return color_map["VS"]  # IT uses VS color
    elif code == "HR":
        return color_map["SP"]  # HR uses SP color
    else:
        # Handle "(all years)" variants like PK_all → PK
        base_code = code.replace("_all", "")
        return color_map.get(base_code, "#999999")  # default gray fallback

def plot_population_data(ax, files):
    """Helper function to plot data on a given axis"""
    global max_time
    # Grey background
    #ax.set_facecolor("#E5E4E2")
    # White dashed grid behind everything
    #ax.grid(True, color='white', linestyle='--', linewidth=0.7, alpha=1)
    #ax.set_axisbelow(True)
    
    for file in files:
        path = f"..//results/demographic_analyses/{file['output_dir']}/posterior_ne.npz"
        
        if not os.path.exists(path):
            missing_files.append(file["name"])
            continue
        
        try:
            data = np.load(path)
            x = data["times"] * 2
            y = np.quantile(data["ne"], 0.5, axis=0)
            y_low = np.quantile(data["ne"], 0.025, axis=0)
            y_high = np.quantile(data["ne"], 0.975, axis=0)
            
            max_time = max(max_time, x[-1])
            
            code = get_population_code(file["name"])
            color = get_color(code)
            
            ax.plot(x, y, lw=2.5, label=file["name"], color=color, zorder=3)
            ax.fill_between(x, y_low, y_high, alpha=0.25, color=color, zorder=2)
            
        except Exception as e:
            print(f"Error loading {file['name']}: {e}")
            missing_files.append(file['name'])


plt.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.5)
# --- Plot all populations ---
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
plot_population_data(ax, metadata)


# --- Formatting ---
ax.set_yscale("log")
ax.set_xscale("symlog")
ax.set_ylabel("Effective population size ($N_e$)", fontsize=8, fontweight='medium', fontproperties=arial_font)
ax.set_xlabel("Time (years ago)", fontsize=8, fontweight='medium', fontproperties=arial_font)

# Legend below the plot
ax.legend(frameon=True, fontsize=9, loc='upper center', framealpha=0.95,
          edgecolor='gray', fancybox=True, ncol=3, bbox_to_anchor=(0.5, -0.15))


ax.set_axisbelow(True)
ax.set_xlim(10, max_time)


# Layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
#plt.savefig("posterior_ne_all_islands.pdf", format="pdf", bbox_inches="tight", dpi=300)
#plt.savefig("posterior_ne_all_islands.svg", format="svg", bbox_inches="tight", dpi=300)

# Missing file report
if missing_files:
    print(f"\nWarning: {len(missing_files)} file(s) not found:")
    for name in missing_files:
        print(f"  - {name}")

plt.show()

```


```python
metadata2 = [
    {"name": "Island PK (2024)", "output_dir": "results/PK_2024"},
    {"name": "Island PM (2024)", "output_dir": "results/PM_2024"},
    {"name": "Island SC (2024)", "output_dir": "results/SC_2024"},
    {"name": "Mainland HR", "output_dir": "results/TR"},
    {"name": "Mainland IT", "output_dir": "results/IT_2024"},
]

```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter

# --- Enhanced Style Configuration ---
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Create figure with single plot
fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)

# Use tab10 palette - excellent for 9 distinguishable colors
colors = sns.color_palette("tab10", n_colors=9)

# Combine all files
all_files = metadata2

max_time = 0
missing_files = []

def plot_population_data(ax, files, colors):
    """Helper function to plot data on a given axis"""
    global max_time
    
    for idx, file in enumerate(files):
        path = f"../results/demographic_analyses/{file['output_dir']}/posterior_ne.npz"
        
        if not os.path.exists(path):
            missing_files.append(file['name'])
            continue
        
        try:
            data = np.load(path)
            x = data["times"] * 2
            y = np.quantile(data["ne"], 0.5, axis=0)
            y_low = np.quantile(data["ne"], 0.025, axis=0)
            y_high = np.quantile(data["ne"], 0.975, axis=0)
            
            max_time = max(max_time, x[-1])
            
            # Plot with distinct colors and improved styling
            color = colors[idx % len(colors)]
            ax.plot(x, y, lw=2.5, label=file["name"], color=color, zorder=3)
            ax.fill_between(x, y_low, y_high, alpha=0.25, color=color, zorder=2)
            
        except Exception as e:
            print(f"Error loading {file['name']}: {e}")
            missing_files.append(file['name'])

# Plot all populations
plot_population_data(ax, all_files, colors)

# Formatting
ax.set_yscale("log")
ax.set_xscale("symlog")
ax.set_ylabel("Effective population size ($N_e$)", fontsize=13, fontweight='medium')
ax.set_xlabel("Time (years ago)", fontsize=13, fontweight='medium')
ax.set_title("Posterior Effective Population Size Through Time", 
             fontsize=14, fontweight='bold', pad=10)

# Legend below the plot
ax.legend(frameon=True, fontsize=9, loc='upper center', framealpha=0.95,
          edgecolor='gray', fancybox=True, ncol=3, bbox_to_anchor=(0.5, -0.15))

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xlim(100, max_time)

# Tighter layout with extra space for legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# --- Save in multiple formats ---
plt.savefig("posterior_ne_2024.pdf", format="pdf", 
            bbox_inches="tight", dpi=300)
plt.savefig("posterior_ne_2024.png", format="png", 
            bbox_inches="tight", dpi=300)

# Report missing files if any
if missing_files:
    print(f"\nWarning: {len(missing_files)} file(s) not found:")
    for name in missing_files:
        print(f"  - {name}")

plt.show()
```

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

#### 9.1.1. Plot GWAS


```python
# Read data
df_SVL = pd.read_csv('../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/svl/GWAS/psiculus_inbreeding_svl_GWAS.assoc.txt', sep='\t')

# ----- Bonferroni threshold -----
n_tests = df_SVL.shape[0]
alpha = 0.05 / n_tests
bonferroni_line = -np.log10(alpha)
```


```python
# Create basic Manhattan plot
plt.figure(figsize=(14, 6))

# Use simple numeric index for x-axis
x_positions = range(len(df_SVL))
p_values = -np.log10(df_SVL['p_lrt'])

# Color by chromosome
unique_chroms = df_SVL['chr'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chroms)))

for i, chrom in enumerate(unique_chroms):
    chrom_mask = df_SVL['chr'] == chrom
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
plt.title('GEMMA LRT P-values SVL')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

#### 9.1.2. Plot genotypes


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

sns.set(style="whitegrid")

# -----------------------------
# 1. Load raw genotypes and phenotype
# -----------------------------
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/svl/GWAS/svl_genotypes_GWAS.raw", delim_whitespace=True)
df["PHENOTYPE"] = df["PHENOTYPE"].replace(-9, np.nan)
df = df.dropna(subset=["PHENOTYPE"])
df.rename(columns={"PHENOTYPE":"phenotype"}, inplace=True)

# -----------------------------
# 2. Load allele lookup table
# -----------------------------
labels = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/svl/psiculus_inbreeding_svl_GWAS.bim", sep=r"\s+",
                     names=["CHR","SNP","CM","POS","A1","A0"])

# -----------------------------
# 3. Map numeric genotypes to allele labels
# -----------------------------
snp_cols = [c for c in df.columns if ":" in c]  # your SNP columns
for col in snp_cols:
    # get SNP name without "_allele" suffix
    snp_name = col.split("_")[0]
    row = labels[labels["SNP"] == snp_name]
    if row.empty:
        raise ValueError(f"Allele info not found for {col}")
    A1 = row["A1"].values[0]
    A0 = row["A0"].values[0]
    df[col + "_label"] = df[col].map({0: f"{A0}/{A0}", 1: f"{A0}/{A1}", 2: f"{A1}/{A1}"}).astype("category")

# -----------------------------
# 4. Extract population codes from IID
# -----------------------------
df['Population'] = df['IID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
df['Population'] = df['Population'].str.replace(r'^24', '', regex=True)
df['Population'] = df['Population'].replace({'CV': 'RG', 'FV': 'RG'})

# -----------------------------
# 5. Define color map
# -----------------------------
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

# -----------------------------
# 6. Create one big multi-plot
# -----------------------------
n = len(snp_cols)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
axes = axes.flatten()

for ax, col in zip(axes, snp_cols):
    sns.boxplot(
        x=col+"_label", y="phenotype", data=df,
        showcaps=True, boxprops={'facecolor': 'None'},
        ax=ax
    )
    sns.stripplot(
        x=col+"_label", y="phenotype", data=df,
        hue="Population", dodge=True, jitter=True, size=4,
        alpha=0.8, palette=color_map, ax=ax
    )
    ax.set_xlabel("Genotype")
    ax.set_ylabel("Phenotype (SVL)")
    ax.set_title(f"{col} vs phenotype")

    # One legend for all subplots → remove here
    ax.legend([], [], frameon=False)

# Add single legend outside the grid
handles, labels_ = ax.get_legend_handles_labels()
fig.legend(handles, labels_, title="Population",
           bbox_to_anchor=(1.02, 0.5), loc="center left")

plt.tight_layout(rect=[0,0,0.92,1])

# Save if wanted
fig.savefig("SVL_9SNPs_3x3_multiplot.png", dpi=300, bbox_inches="tight")

plt.show()
```

#### 9.2.1. Plot GWAS 


```python
df_BF = pd.read_csv('../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/bf/GWAS/psiculus_inbreeding_bf_GWAS.assoc.txt', sep='\t')

# ----- Bonferroni threshold -----
n_tests = df_BF.shape[0]
alpha = 0.05 / n_tests
bonferroni_line = -np.log10(alpha)
```


```python
# Create basic Manhattan plot
plt.figure(figsize=(14, 6))

# Use simple numeric index for x-axis
x_positions = range(len(df_BF))
p_values = -np.log10(df_BF['p_lrt'])

# Color by chromosome
unique_chroms = df_BF['chr'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chroms)))

for i, chrom in enumerate(unique_chroms):
    chrom_mask = df_BF['chr'] == chrom
    plt.scatter(np.array(x_positions)[chrom_mask], 
                p_values[chrom_mask], 
                color=colors[i], 
                s=10, 
                alpha=0.7,
                label=str(chrom))


plt.axhline(y=bonferroni_line, color='red', linestyle='--', linewidth=1,
            label=f'Bonferroni (0.05/{n_tests:,} = {alpha:.2e})')
plt.axhline(y=-np.log10(5e-8), color='grey', linestyle='--', linewidth=1, label='Threshold 5e-8')

plt.xlabel('SNP Position')
plt.ylabel('-log10(P-value) - Likelihood Ratio Test')
plt.title('GEMMA LRT P-values Bite Force')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```

#### 9.2.2. Plot genotypes


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

sns.set(style="whitegrid")

# -----------------------------
# 1. Load raw genotypes and phenotype
# -----------------------------
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/bf/GWAS/bf_genotypes_GWAS.raw", delim_whitespace=True)
df["PHENOTYPE"] = df["PHENOTYPE"].replace(-9, np.nan)
df = df.dropna(subset=["PHENOTYPE"])
df.rename(columns={"PHENOTYPE":"phenotype"}, inplace=True)

# -----------------------------
# 2. Load allele lookup table
# -----------------------------
labels = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/bf/GWAS/bf_genotypes_GWAS_labelstop.txt", sep=r"\s+",
                     names=["CHR","SNP","CM","POS","A1","A0"])

# -----------------------------
# 3. Map numeric genotypes to allele labels
# -----------------------------
snp_cols = [c for c in df.columns if ":" in c]  # your SNP columns
for col in snp_cols:
    # get SNP name without "_allele" suffix
    snp_name = col.split("_")[0]
    row = labels[labels["SNP"] == snp_name]
    if row.empty:
        raise ValueError(f"Allele info not found for {col}")
    A1 = row["A1"].values[0]
    A0 = row["A0"].values[0]
    df[col + "_label"] = df[col].map({0: f"{A0}/{A0}", 1: f"{A0}/{A1}", 2: f"{A1}/{A1}"}).astype("category")

# -----------------------------
# 4. Extract population codes from IID
# -----------------------------
df['Population'] = df['IID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
df['Population'] = df['Population'].str.replace(r'^24', '', regex=True)

# -----------------------------
# 5. Define color map
# -----------------------------
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

# -----------------------------
# 6. Plot SNPs vs phenotype colored by population
# -----------------------------
for col in snp_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=col+"_label", y="phenotype", data=df, showcaps=True, boxprops={'facecolor':'None'})
    sns.stripplot(x=col+"_label", y="phenotype", data=df, hue="Population",
                  dodge=True, jitter=True, size=6, alpha=0.8, palette=color_map)
    plt.xlabel("Genotype")
    plt.ylabel("Phenotype (Bite Force)")
    plt.title(f"{col} genotype vs phenotype")
    plt.legend(title="Population", bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    
    # Save figure
    #plt.savefig(f"{col}_genotype_vs_phenotype.png", dpi=300, bbox_inches='tight')
    plt.show()
```


```python
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm

# -----------------------------
# 1. Load raw genotypes and phenotype
# -----------------------------
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/bf/GWAS/bf_genotypes_GWAS.raw", delim_whitespace=True)
df["PHENOTYPE"] = df["PHENOTYPE"].replace(-9, np.nan)
df = df.dropna(subset=["PHENOTYPE"])
df.rename(columns={"PHENOTYPE":"phenotype"}, inplace=True)

# -----------------------------
# 2. Load allele lookup table
# -----------------------------
labels = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/bf/GWAS/bf_genotypes_GWAS_labelstop.txt", sep=r"\s+",
                     names=["CHR","SNP","CM","POS","A1","A0"])

# -----------------------------
# 3. Extract SNP columns
# -----------------------------
snp_cols = [c for c in df.columns if ":" in c]

# -----------------------------
# 4. Map numeric genotypes to allele labels and keep numeric for regression
# -----------------------------
for col in snp_cols:
    snp_name = col.split("_")[0]
    row = labels[labels["SNP"] == snp_name]
    if row.empty:
        raise ValueError(f"Allele info not found for {col}")
    A1 = row["A1"].values[0]
    A0 = row["A0"].values[0]
    df[col + "_label"] = df[col].map({0: f"{A0}/{A0}", 1: f"{A0}/{A1}", 2: f"{A1}/{A1}"}).astype("category")
    df[col + "_num"] = df[col]  # numeric genotype for regression

# -----------------------------
# 5. Extract population codes from IID
# -----------------------------
df['Population'] = df['IID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
df['Population'] = df['Population'].str.replace(r'^24', '', regex=True)

# -----------------------------
# 6. Run regression per SNP per population
# -----------------------------
results = []

for col in snp_cols:
    for pop in df['Population'].unique():
        pop_df = df[df['Population'] == pop].dropna(subset=['phenotype', col+"_num"])
        
        if len(pop_df) < 3:
            continue  # skip if too few samples
        
        X = sm.add_constant(pop_df[col+"_num"])  # intercept + genotype
        y = pop_df['phenotype']
        
        model = sm.OLS(y, X).fit()
        
        results.append({
            "SNP": col,
            "Population": pop,
            "N": len(pop_df),
            "Beta": model.params[col+"_num"],
            "SE": model.bse[col+"_num"],
            "t": model.tvalues[col+"_num"],
            "p_value": model.pvalues[col+"_num"],
            "R2": model.rsquared
        })

# -----------------------------
# 7. Combine results into a table
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["SNP","Population"])
results_df.reset_index(drop=True, inplace=True)

results_df
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

# Loop over SNPs
for col in snp_cols:
    plt.figure(figsize=(7,5))
    
    # Boxplot per genotype, split by population
    sns.boxplot(x=col+"_num", y="phenotype", hue="Population", data=df,
                palette=color_map, showcaps=True, fliersize=0, width=0.6)
    
    # Jittered points on top of the boxplot
    for i, pop in enumerate(df['Population'].unique()):
        pop_df = df[df['Population']==pop].dropna(subset=[col+"_num","phenotype"])
        if len(pop_df) == 0:
            continue
        # small jitter for points
        jitter = np.random.uniform(-0.15, 0.15, size=len(pop_df))
        plt.scatter(pop_df[col+"_num"] + jitter, pop_df['phenotype'],
                    color=color_map.get(pop,'grey'), s=60, alpha=0.7, label=None)
        
        # Regression lines
        if len(pop_df) >= 3:
            sns.regplot(x=col+"_num", y="phenotype", data=pop_df,
                        scatter=False, color=color_map.get(pop,'grey'), ci=None)
            
            # Add p-value from previous regression results
            res_row = results_df[(results_df['SNP']==col) & (results_df['Population']==pop)]
            if not res_row.empty:
                pval = res_row['p_value'].values[0]
                # Annotate p-value above the boxplot for that population
                x_pos = pop_df[col+"_num"].mean()
                y_pos = pop_df['phenotype'].max() + 0.2
                plt.text(x_pos, y_pos, f"p={pval:.3g}", color=color_map.get(pop,'grey'), fontsize=10, ha='center')
    
    plt.xlabel("Genotype (0=hom ref, 1=het, 2=hom alt)")
    plt.ylabel("Phenotype (Bite Force)")
    plt.title(f"{col} genotype vs phenotype with regression per population")
    plt.legend(title="Population", bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    
    # Optional: save figure
    # plt.savefig(f"{col}_box_regression_population.png", dpi=300, bbox_inches='tight')
    
    plt.show()

```

#### 9.3.1. Plot GWAS


```python
# Read data
df_SP = pd.read_csv('../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/sprint/GWAS/psiculus_inbreeding_sprint_GWAS.assoc.txt', sep='\t')
# ----- Bonferroni threshold -----
n_tests = df_SP.shape[0]
alpha = 0.05 / n_tests
bonferroni_line = -np.log10(alpha)
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Create basic Manhattan plot
plt.figure(figsize=(14, 6))

# Use simple numeric index for x-axis
x_positions = range(len(df_SP))
p_values = -np.log10(df_SP['p_lrt'])

# Color by chromosome
unique_chroms = df_SP['chr'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chroms)))

for i, chrom in enumerate(unique_chroms):
    chrom_mask = df_SP['chr'] == chrom
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
plt.title('GEMMA LRT P-values SPRINT')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

#### 9.3.2. Plot genotypes


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

sns.set(style="whitegrid")

# -----------------------------
# 1. Load raw genotypes and phenotype
# -----------------------------
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/sprint/GWAS/sprint_genotypes_GWAS.raw", delim_whitespace=True)
df["PHENOTYPE"] = df["PHENOTYPE"].replace(-9, np.nan)
df = df.dropna(subset=["PHENOTYPE"])
df.rename(columns={"PHENOTYPE":"phenotype"}, inplace=True)

# -----------------------------
# 2. Load allele lookup table
# -----------------------------
labels = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/psiculus_inbreeding_performance_GWAS.bim", sep=r"\s+",
                     names=["CHR","SNP","CM","POS","A1","A0"])

# -----------------------------
# 3. Map numeric genotypes to allele labels
# -----------------------------
snp_cols = [c for c in df.columns if ":" in c]  # your SNP columns
for col in snp_cols:
    # get SNP name without "_allele" suffix
    snp_name = col.split("_")[0]
    row = labels[labels["SNP"] == snp_name]
    if row.empty:
        raise ValueError(f"Allele info not found for {col}")
    A1 = row["A1"].values[0]
    A0 = row["A0"].values[0]
    df[col + "_label"] = df[col].map({0: f"{A0}/{A0}", 1: f"{A0}/{A1}", 2: f"{A1}/{A1}"}).astype("category")

# -----------------------------
# 4. Extract population codes from IID
# -----------------------------
df['Population'] = df['IID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
df['Population'] = df['Population'].str.replace(r'^24', '', regex=True)
df['Population'] = df['Population'].replace({'CV': 'RG', 'FV': 'RG'})

# -----------------------------
# 5. Define color map
# -----------------------------
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

# -----------------------------
# 6. Create one big multi-plot
# -----------------------------
n = len(snp_cols)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
axes = axes.flatten()

for ax, col in zip(axes, snp_cols):
    sns.boxplot(
        x=col+"_label", y="phenotype", data=df,
        showcaps=True, boxprops={'facecolor': 'None'},
        ax=ax
    )
    sns.stripplot(
        x=col+"_label", y="phenotype", data=df,
        hue="Population", dodge=True, jitter=True, size=4,
        alpha=0.8, palette=color_map, ax=ax
    )
    ax.set_xlabel("Genotype")
    ax.set_ylabel("Phenotype (sprint)")
    ax.set_title(f"{col} vs phenotype")

    # One legend for all subplots → remove here
    ax.legend([], [], frameon=False)

# Add single legend outside the grid
handles, labels_ = ax.get_legend_handles_labels()
fig.legend(handles, labels_, title="Population",
           bbox_to_anchor=(1.02, 0.5), loc="center left")

plt.tight_layout(rect=[0,0,0.92,1])

# Save if wanted
fig.savefig("sprint_9SNPs_3x3_multiplot.png", dpi=300, bbox_inches="tight")

plt.show()
```


```python
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm

# -----------------------------
# 1. Load raw genotypes and phenotype
# -----------------------------
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/sprint/GWAS/sprint_genotypes_GWAS.raw", delim_whitespace=True)
df["PHENOTYPE"] = df["PHENOTYPE"].replace(-9, np.nan)
df = df.dropna(subset=["PHENOTYPE"])
df.rename(columns={"PHENOTYPE":"phenotype"}, inplace=True)

# -----------------------------
# 2. Load allele lookup table
# -----------------------------
labels = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/psiculus_inbreeding_performance_GWAS.bim", sep=r"\s+",
                     names=["CHR","SNP","CM","POS","A1","A0"])

# -----------------------------
# 3. Extract SNP columns
# -----------------------------
snp_cols = [c for c in df.columns if ":" in c]

# -----------------------------
# 4. Map numeric genotypes to allele labels and keep numeric for regression
# -----------------------------
for col in snp_cols:
    snp_name = col.split("_")[0]
    row = labels[labels["SNP"] == snp_name]
    if row.empty:
        raise ValueError(f"Allele info not found for {col}")
    A1 = row["A1"].values[0]
    A0 = row["A0"].values[0]
    df[col + "_label"] = df[col].map({0: f"{A0}/{A0}", 1: f"{A0}/{A1}", 2: f"{A1}/{A1}"}).astype("category")
    df[col + "_num"] = df[col]  # numeric genotype for regression

# -----------------------------
# 5. Extract population codes from IID
# -----------------------------
df['Population'] = df['IID'].apply(lambda x: ''.join(re.findall(r'[A-Z]+', x)))
df['Population'] = df['Population'].str.replace(r'^24', '', regex=True)

# -----------------------------
# 6. Run regression per SNP per population
# -----------------------------
results = []

for col in snp_cols:
    for pop in df['Population'].unique():
        pop_df = df[df['Population'] == pop].dropna(subset=['phenotype', col+"_num"])
        
        if len(pop_df) < 3:
            continue  # skip if too few samples
        
        X = sm.add_constant(pop_df[col+"_num"])  # intercept + genotype
        y = pop_df['phenotype']
        
        model = sm.OLS(y, X).fit()
        
        results.append({
            "SNP": col,
            "Population": pop,
            "N": len(pop_df),
            "Beta": model.params[col+"_num"],
            "SE": model.bse[col+"_num"],
            "t": model.tvalues[col+"_num"],
            "p_value": model.pvalues[col+"_num"],
            "R2": model.rsquared
        })

# -----------------------------
# 7. Combine results into a table
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["SNP","Population"])
results_df.reset_index(drop=True, inplace=True)

results_df
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

# Loop over SNPs
for col in snp_cols:
    plt.figure(figsize=(7,5))
    
    # Boxplot per genotype, split by population
    sns.boxplot(x=col+"_num", y="phenotype", hue="Population", data=df,
                palette=color_map, showcaps=True, fliersize=0, width=0.6)
    
    # Jittered points on top of the boxplot
    for i, pop in enumerate(df['Population'].unique()):
        pop_df = df[df['Population']==pop].dropna(subset=[col+"_num","phenotype"])
        if len(pop_df) == 0:
            continue
        # small jitter for points
        jitter = np.random.uniform(-0.15, 0.15, size=len(pop_df))
        plt.scatter(pop_df[col+"_num"] + jitter, pop_df['phenotype'],
                    color=color_map.get(pop,'grey'), s=60, alpha=0.7, label=None)
        
        # Regression lines
        if len(pop_df) >= 3:
            sns.regplot(x=col+"_num", y="phenotype", data=pop_df,
                        scatter=False, color=color_map.get(pop,'grey'), ci=None)
            
            # Add p-value from previous regression results
            res_row = results_df[(results_df['SNP']==col) & (results_df['Population']==pop)]
            if not res_row.empty:
                pval = res_row['p_value'].values[0]
                # Annotate p-value above the boxplot for that population
                x_pos = pop_df[col+"_num"].mean()
                y_pos = pop_df['phenotype'].max() + 0.2
                plt.text(x_pos, y_pos, f"p={pval:.3g}", color=color_map.get(pop,'grey'), fontsize=10, ha='center')
    
    plt.xlabel("Genotype (0=hom ref, 1=het, 2=hom alt)")
    plt.ylabel("Phenotype (Sprint)")
    plt.title(f"{col} genotype vs phenotype with regression per population")
    plt.legend(title="Population", bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    
    # Optional: save figure
    # plt.savefig(f"{col}_box_regression_population.png", dpi=300, bbox_inches='tight')
    
    plt.show()

```

#### 9.4.1. Plot GWAS


```python
# Read data
df_SPC = pd.read_csv('../results/population_analyses/rPodSic1.hap1.1/genotype_phenotype/performance/spermc/GWAS/psiculus_inbreeding_spermc_GWAS.assoc.txt', sep='\t')
# ----- Bonferroni threshold -----
n_tests = df_SPC.shape[0]
alpha = 0.05 / n_tests
bonferroni_line = -np.log10(alpha)
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

sns.set(style="whitegrid")

# --- Load ALL GWAS chromosome files ---
gwas_files = sorted(glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_ROHstatus/roh_gwas_*_ROHstatus.tsv"))

if len(gwas_files) == 0:
    raise FileNotFoundError("No GWAS files found!")

df_list = []
for f in gwas_files:
    tmp = pd.read_csv(f, sep="\t")
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

print(f"Loaded {len(gwas_files)} chromosomes")
print(f"Total SNPs: {len(df)}")

```


```python
# --- Parse SNP info ---
df[['chrom', 'pos']] = df['SNP_ID'].str.split(':', expand=True)
df['pos'] = pd.to_numeric(df['pos'], errors='coerce')

# Convert stats to numeric
df['ROH_status_pval'] = pd.to_numeric(df['ROH_status_pval'], errors='coerce')
df['ROH_status_coef'] = pd.to_numeric(df['ROH_status_coef'], errors='coerce')
df['ROH_status_log10p'] = -np.log10(df['ROH_status_pval'])

# Drop incomplete rows
df = df.dropna(subset=['chrom', 'pos', 'ROH_status_log10p'])

# --- Chromosome ordering ---
chrom_order = (
    df[['chrom']]
    .drop_duplicates()
    .sort_values(by='chrom')
    ['chrom']
    .tolist()
)

df['chrom'] = pd.Categorical(df['chrom'], categories=chrom_order, ordered=True)
df = df.sort_values(['chrom', 'pos'])

# --- Create cumulative genomic positions ---
chrom_sizes = df.groupby('chrom')['pos'].max()
chrom_offsets = chrom_sizes.cumsum() - chrom_sizes

df['cum_pos'] = df.apply(
    lambda r: r['pos'] + chrom_offsets[r['chrom']],
    axis=1
)

# --- Bonferroni threshold ---
bonf_thresh = 0.05 / len(df)
bonf_line = -np.log10(bonf_thresh)

# --- X-axis chromosome centers ---
chrom_centers = chrom_offsets + chrom_sizes / 2

```


```python
# --- Plot ---
plt.figure(figsize=(18, 6))

colors = ['#4daf4a', '#377eb8']

for i, chrom in enumerate(chrom_order):
    d = df[df['chrom'] == chrom]
    plt.scatter(
        d['cum_pos'],
        d['ROH_status_log10p'],
        s=6,
        color=colors[i % 2],
        alpha=0.7
    )

plt.axhline(bonf_line, color='red', linestyle='--', label='Bonferroni')

plt.xticks(chrom_centers, chrom_order, rotation=90)
plt.xlabel("Chromosome")
plt.ylabel("-log10(p-value)")
plt.title("Genome-wide Manhattan plot: BF ~ ROH status")
plt.legend()
plt.tight_layout()
plt.show()
```


```python
# --- QQ plot ---
plt.figure(figsize=(6,6))

for pred in ['ROH_status']:
    pvals = df[f'{pred}_pval'].dropna()
    expected = -np.log10(np.linspace(1/len(pvals), 1, len(pvals)))
    observed = -np.log10(np.sort(pvals))
    plt.scatter(expected, observed, s=5, alpha=0.6, label=pred)

plt.plot([0, max(expected)], [0, max(expected)], color='red', linestyle='--')
plt.xlabel('Expected -log10(p)')
plt.ylabel('Observed -log10(p)')
plt.title('QQ Plot: ROH_status')
plt.legend()
plt.show()
```


```python
alpha = 0.05
bonf_thresh = alpha / len(df)

print(f"Bonferroni p-value threshold: {bonf_thresh:.2e}")

sig_snps = df[
    (df['ROH_status_pval'] > 0) &                      # exclude p = 0
    (np.isfinite(df['ROH_status_log10p'])) &            # exclude inf / NaN
    (df['ROH_status_pval'] <= bonf_thresh)
].copy()

print(f"Number of Bonferroni-significant SNPs: {len(sig_snps)}")

```


```python
sig_snps = sig_snps[[
    'chrom',
    'pos',
    'SNP_ID',
    'ROH_status_coef',
    'ROH_status_pval',
    'ROH_status_log10p',
    'cum_pos'
]].sort_values('ROH_status_pval')
out_file = "bonferroni_significant_SNPs_ROHstatus_BF.tsv"
sig_snps.to_csv(out_file, sep="\t", index=False)

print(f"Saved significant SNPs to: {out_file}")
```

#### 10.2.1. Effect sizes significant SNPs


```python
plt.figure(figsize=(14,5))

plt.scatter(
    sig_snps['cum_pos'],
    sig_snps['ROH_status_coef'],
    c=np.sign(sig_snps['ROH_status_coef']),
    cmap='bwr',
    s=40
)

plt.axhline(0, color='black', linestyle='--')

plt.xlabel('Genomic position')
plt.ylabel('Effect size (β)')
plt.title('Effect sizes of Bonferroni-significant ROH SNPs')

plt.tight_layout()
plt.show()

```


```python
alpha = 0.05
bonf_thresh = alpha / len(df)

# Only finite, non-zero p-values count as significant
df['bonf_sig'] = (
    (df['ROH_status_pval'] > 0) &                      # exclude p = 0
    (np.isfinite(df['ROH_status_log10p'])) &           # exclude inf or NaN
    (df['ROH_status_pval'] <= bonf_thresh)            # Bonferroni threshold
)

print(f"Number of Bonferroni-significant SNPs: {df['bonf_sig'].sum()}")

```


```python
colors = ['#4daf4a', '#377eb8']

plt.figure(figsize=(18, 6))

# --- Background: all SNPs, alternating by chromosome ---
for i, chrom in enumerate(df['chrom'].cat.categories):
    d = df[df['chrom'] == chrom]
    plt.scatter(
        d['cum_pos'],
        d['ROH_status_coef'],
        s=6,
        alpha=0.5,
        color=colors[i % 2]
    )

# --- Significant SNPs: negative effects (red) ---
neg_sig = df['bonf_sig'] & (df['ROH_status_coef'] < 0)
plt.scatter(
    df.loc[neg_sig, 'cum_pos'],
    df.loc[neg_sig, 'ROH_status_coef'],
    s=35,
    color='red',
    edgecolor='black',
    label='Bonferroni sig (β < 0)'
)

# --- Significant SNPs: positive effects (purple) ---
pos_sig = df['bonf_sig'] & (df['ROH_status_coef'] > 0)
plt.scatter(
    df.loc[pos_sig, 'cum_pos'],
    df.loc[pos_sig, 'ROH_status_coef'],
    s=35,
    color='purple',
    edgecolor='black',
    label='Bonferroni sig (β > 0)'
)

# Reference line
plt.axhline(0, color='black', linestyle='--')

plt.xlabel('Chromosome')
plt.ylabel('Effect size (β)')
plt.title('Effect-size Manhattan plot: BF ~ ROH status')

plt.legend()
plt.tight_layout()
plt.show()

```

#### 10.2.2. ROH status - phenotype


```python
import pandas as pd
import os

# --- Paths ---
INPUT_DIR = "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/roh_gwas_input/"
SIG_SNP_FILE = "bonferroni_significant_SNPs_ROHstatus_BF.tsv"
OUTPUT_FILE = "ROH_status_per_individual_all_significant_SNPs_.tsv"

# --- Load Bonferroni-significant SNPs ---
sig_snps = pd.read_csv(SIG_SNP_FILE, sep="\t")
print(f"Loaded {len(sig_snps)} Bonferroni-significant SNPs")

# --- Map SNP → effect size (coefficient) ---
beta_map = sig_snps.set_index('SNP_ID')['ROH_status_coef']

# --- Group SNPs by chromosome for fast lookup ---
snps_by_chrom = {chrom: set(sdf['SNP_ID']) for chrom, sdf in sig_snps.groupby('chrom')}
print("Significant SNPs per chromosome:")
for chrom, snps in snps_by_chrom.items():
    print(f"{chrom}: {len(snps)}")

# --- Remove old output if exists ---
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print("Existing output file removed")

# --- Stream through each chromosome and write filtered rows ---
first_write = True
CHUNKSIZE = 1_000_000  # increase if you have enough RAM

for chrom, snp_set in snps_by_chrom.items():

    input_file = os.path.join(INPUT_DIR, f"gwas_input_{chrom}.tsv")
    if not os.path.exists(input_file):
        print(f"⚠️ Input file missing for {chrom}, skipping")
        continue

    print(f"Processing {chrom} ({len(snp_set)} SNPs)")

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNKSIZE, dtype=str):
        # Keep only SNPs in our significant set and make a copy
        filtered_chunk = chunk.loc[chunk['SNP_ID'].isin(snp_set)].copy()
        if not filtered_chunk.empty:
            # Add coefficient
            filtered_chunk['ROH_effect_beta'] = filtered_chunk['SNP_ID'].map(beta_map)
            # Write chunk directly
            filtered_chunk.to_csv(
                OUTPUT_FILE,
                sep="\t",
                mode='a',
                index=False,
                header=first_write
            )
            first_write = False  # only write header once

```


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import os

sns.set(style="whitegrid")

# --- Paths ---
INPUT_FILE = "ROH_status_per_individual_all_significant_SNPs.tsv"
OUT_DIR = "roh_status_plots_per_snp"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(INPUT_FILE, sep="\t")

# --- Clean phenotype ---
df["PHENO"] = pd.to_numeric(df["PHENO"], errors="coerce")
df = df.dropna(subset=["PHENO"])
df.rename(columns={"PHENO": "phenotype"}, inplace=True)

# --- Ensure ROH_status is categorical and ordered ---
df["ROH_status"] = df["ROH_status"].astype(int)
df["ROH_status"] = pd.Categorical(df["ROH_status"], categories=[0, 1], ordered=True)

# --- Extract population from IID ---
df["Population"] = df["IID"].str.extract(r'([A-Z]+)')
df["Population"] = df["Population"].str.replace(r'^24', '', regex=True)

# --- Population color map ---
color_map = {
    'PJ':'#6BAED6', 'MP': '#2171B5', 'KP': '#4292C6', 'PM': '#08306B',
    'BJ':'#08519C', 'SC':'#9ECAE1', 'PK':'#C6DBEF', 'VS':'#238B45',
    'TM':'#00441B', 'RG':'#BAE4B3', 'SP':'#74C476', 'TR':'#006D2C',
    'PL':'#41AB5D'
}

# --- Get unique SNPs ---
snps = df["SNP_ID"].unique()
print(f"Plotting {len(snps)} significant SNPs")

# --- Plot per SNP ---
for snp in snps:
    df_snp = df[df["SNP_ID"] == snp]

    if df_snp["ROH_status"].nunique() < 2:
        # Skip SNPs with no variation in ROH status
        continue

    beta = df_snp["ROH_effect_beta"].iloc[0]

    plt.figure(figsize=(6, 4))

    # Boxplot
    sns.boxplot(
        x="ROH_status",
        y="phenotype",
        data=df_snp,
        showcaps=True,
        boxprops={'facecolor': 'none'},
        width=0.5
    )

    # Population-colored points
    sns.stripplot(
        x="ROH_status",
        y="phenotype",
        data=df_snp,
        hue="Population",
        dodge=True,
        jitter=True,
        size=6,
        alpha=0.8,
        palette=color_map
    )

    plt.xlabel("ROH status (0 = no ROH, 1 = ROH)")
    plt.ylabel("Phenotype")
    plt.title(f"{snp}\nROH effect β = {beta:.3f}")

    plt.legend(
        title="Population",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # --- Save instead of show (much faster for many SNPs) ---
    out_file = os.path.join(OUT_DIR, f"{snp.replace(':', '_')}_ROH_status_.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

print("All plots saved.")

```


```python

```


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

#### 11.1.3. All 


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

#### 11.1.4. Plots


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
    plt.rcParams["font.family"] = arial_font.get_name()

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
        linewidth=1,
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
    ax.set_xticklabels(chrom_labels, fontsize=6)

    ax.tick_params(axis="y", labelsize=6)

    ax.set_xlabel("Chromosome", fontsize=8, fontproperties=arial_font)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=8, fontproperties=arial_font)
    ax.set_title(title)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.minorticks_on()
    ax.grid(
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=0.3,
        color="grey",
        alpha=0.5
    )

    # Start dots at y=0
    ax.set_ylim(bottom=0)
    ax.margins(y=0, x=0.01)

    # Black box
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)

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
    fig_width_cm=12,
    fig_height_cm=6,
    out_file="ROH_GWAS_ML_manhattan.png"
)

```


```python
def plot_manhattan_plot(
    file_glob,
    title,
    neff_genome,
    sig_file=None,
    sig_effect_col="BETA_ROH_STATUS",
    chrom_colors=("#208438", "#479BC9"),  # <- MATCHES your palette
    fig_width_cm=6.75,
    fig_height_cm=6,
    point_size=6,
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    out_file=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import glob
    import os

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
    fig, ax = plt.subplots(figsize=(fig_width_cm * cm, fig_height_cm * cm), dpi=300)

    # =========================
    # Load data
    # =========================
    files = sorted(glob.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {file_glob}")

    df = pd.concat((pd.read_csv(f, sep="\t") for f in files), ignore_index=True)

    df["POS"] = df["WINDOW_ID"].apply(window_midpoint)
    df["LOGP"] = -np.log10(df["P_ROH_STATUS"])

    # =========================
    # Chromosome positions
    # =========================
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

    # =========================
    # Threshold
    # =========================
    eff_bonf_logp = -np.log10(0.05 / neff_genome)

    # =========================
    # Grid (IDENTICAL STYLE)
    # =========================
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

    # =========================
    # Scatter
    # =========================
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

    # Threshold line
    ax.axhline(eff_bonf_logp, linestyle="--", linewidth=0.5, color="black")

    # =========================
    # Significant overlay
    # =========================
    if sig_file is not None:

        if isinstance(sig_file, str):
            sig_df = pd.read_csv(sig_file, sep=None, engine="python")
        else:
            sig_df = sig_file.copy()

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
    # =========================
    # Axes styling (MATCHED)
    # =========================
    ax.set_xlabel("")
    ax.set_ylabel(r"$-\log_{10}(p)$")

    ax.set_xticks(chrom_centers)
    ax.set_xticklabels([""] * len(chrom_centers))

    # remove y labels like your stripplot
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelleft=False)

    ax.tick_params(axis='x', rotation=45, pad=4)
    ax.tick_params(axis='y', pad=4)

    # limits
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)

    # =========================
    # Layout (CRUCIAL MATCH)
    # =========================
    ax.set_position([0.10, 0.18, 0.74, 0.74])

    # =========================
    # Save
    # =========================
    fig.canvas.draw()

    if out_file is not None:
        plt.savefig(out_file, dpi=300)
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
    chrom_colors=("#208438", "#145022"),
    fig_width_cm=12,
    fig_height_cm=6,
    out_file="ROH_GWAS_ML_manhattan_test.png"
)
```


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
plot_manhattan_plot(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_ISLAND_*.tsv",
    title="",
    sig_file=sig_df_IL,
    neff_genome=1409,
    chrom_colors=("#479BC9", "#255E7D"),
    fig_width_cm=12,
    fig_height_cm=6,
    out_file="ROH_GWAS_IL_manhattan_test.png"
)

```


```python
def plot_effect_size_bellcurve_filled_with_sig_points(
    file_glob,
    sig_df,
    effect_col="BETA_ROH_STATUS",
    fig_width_cm=10,
    fig_height_cm=10,
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    neg_color="#2C7BB6",
    pos_color="#D7191C",
    curve_alpha=0.5,
    point_size=20,
    out_file=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import glob
    from scipy.stats import gaussian_kde

    # ==================================================
    # Font: Arial (HPC-safe)
    # ==================================================
    arial_font = fm.FontProperties(fname=arial_path)
    plt.rcParams["font.family"] = arial_font.get_name()
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # ==================================================
    # Load GWAS windows via glob (for KDE)
    # ==================================================
    files = sorted(glob.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {file_glob}")

    df_all = pd.concat(
        (pd.read_csv(f, sep="\t") for f in files),
        ignore_index=True
    )

    if effect_col not in df_all.columns:
        raise KeyError(
            f"Column '{effect_col}' not found in GWAS files.\n"
            f"Available columns:\n{list(df_all.columns)}"
        )

    effects_all = df_all[effect_col].dropna().values
    if len(effects_all) < 2:
        raise ValueError("Need at least 2 effect sizes for KDE.")

    # ==================================================
    # KDE (single bell curve)
    # ==================================================
    kde = gaussian_kde(effects_all)
    x = np.linspace(
        effects_all.min() * 1.1,
        effects_all.max() * 1.1,
        1000
    )
    y = kde(x)

    # Split for coloring
    x_neg, y_neg = x[x <= 0], y[x <= 0]
    x_pos, y_pos = x[x >= 0], y[x >= 0]

    # ==================================================
    # Prepare significant points
    # ==================================================
    if effect_col not in sig_df.columns:
        raise KeyError(
            f"Column '{effect_col}' not found in sig_df.\n"
            f"Available columns:\n{list(sig_df.columns)}"
        )

    sig_effects = sig_df[effect_col].dropna().values
    pos = sig_effects > 0
    neg = sig_effects < 0

    # small vertical jitter so points sit on baseline
    y_jitter = np.random.uniform(
        low=0,
        high=y.max() * 0.04,
        size=len(sig_effects)
    )

    # ==================================================
    # Figure
    # ==================================================
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(fig_width_cm * cm, fig_height_cm * cm)
    )

    # Filled KDE
    ax.fill_between(x_neg, y_neg, color=neg_color, alpha=curve_alpha)
    ax.fill_between(x_pos, y_pos, color=pos_color, alpha=curve_alpha)

    # KDE outline
    ax.plot(x, y, color="black", linewidth=1)

    # Zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    # ==================================================
    # Overlay significant points
    # ==================================================
    ax.scatter(
        sig_effects[pos],
        y_jitter[pos],
        color=pos_color,
        s=point_size,
        edgecolor=pos_color,
        linewidth=0.3,
        zorder=3,
        label="Positive effect"
    )

    ax.scatter(
        sig_effects[neg],
        y_jitter[neg],
        color=neg_color,
        s=point_size,
        edgecolor=neg_color,
        linewidth=0.3,
        zorder=3,
        label="Negative effect"
    )

    # ==================================================
    # Axes formatting
    # ==================================================
    ax.set_xlabel("Effect size", fontsize=8, fontproperties=arial_font)
    ax.set_ylabel("Density", fontsize=8, fontproperties=arial_font)
    ax.tick_params(axis="both", labelsize=6)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)
    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.minorticks_on()
    ax.grid(
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=0.3,
        color="grey",
        alpha=0.5
    )

    # Start dots at y=0
    #ax.set_ylim(bottom=0.00000001)
    #ax.margins(y=0.01, x=0.01)

    #ax.legend(frameon=False, fontsize=6)

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
def plot_effect_size_bellcurve_filled_with_sig_points(
    file_glob,
    sig_df,
    effect_col="BETA_ROH_STATUS",
    fig_width_cm=10,
    fig_height_cm=10,
    arial_path="/scratch/antwerpen/206/vsc20666/arial.ttf",
    neg_color="#2C7BB6",
    pos_color="#D7191C",
    curve_alpha=0.5,
    point_size=20,
    out_file=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import glob
    from scipy.stats import gaussian_kde

    # ==================================================
    # Font: Arial (HPC-safe)
    # ==================================================
    arial_font = fm.FontProperties(fname=arial_path)
    plt.rcParams["font.family"] = arial_font.get_name()
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # ==================================================
    # Load GWAS windows via glob (for KDE)
    # ==================================================
    files = sorted(glob.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {file_glob}")

    df_all = pd.concat(
        (pd.read_csv(f, sep="\t") for f in files),
        ignore_index=True
    )

    if effect_col not in df_all.columns:
        raise KeyError(
            f"Column '{effect_col}' not found in GWAS files.\n"
            f"Available columns:\n{list(df_all.columns)}"
        )

    effects_all = df_all[effect_col].dropna().values
    if len(effects_all) < 2:
        raise ValueError("Need at least 2 effect sizes for KDE.")

    # ==================================================
    # KDE (single bell curve)
    # ==================================================
    kde = gaussian_kde(effects_all)
    x = np.linspace(
        effects_all.min() * 1.1,
        effects_all.max() * 1.1,
        1000
    )
    y = kde(x)

    # Split for coloring
    x_neg, y_neg = x[x <= 0], y[x <= 0]
    x_pos, y_pos = x[x >= 0], y[x >= 0]

    # ==================================================
    # Prepare significant points
    # ==================================================
    if effect_col not in sig_df.columns:
        raise KeyError(
            f"Column '{effect_col}' not found in sig_df.\n"
            f"Available columns:\n{list(sig_df.columns)}"
        )

    sig_effects = sig_df[effect_col].dropna().values
    pos = sig_effects > 0
    neg = sig_effects < 0

    # small vertical jitter so points sit on baseline
    y_jitter = np.random.uniform(
        low=0,
        high=y.max() * 0.04,
        size=len(sig_effects)
    )

    # ==================================================
    # Figure
    # ==================================================
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(fig_width_cm * cm, fig_height_cm * cm)
    )

    # Filled KDE
    ax.fill_between(x_neg, y_neg, color=neg_color, alpha=curve_alpha)
    ax.fill_between(x_pos, y_pos, color=pos_color, alpha=curve_alpha)

    # KDE outline
    ax.plot(x, y, color="black", linewidth=1)

    # Zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    # ==================================================
    # Overlay significant points
    # ==================================================
    ax.scatter(
        sig_effects[pos],
        y_jitter[pos],
        color=pos_color,
        s=point_size,
        edgecolor=pos_color,
        linewidth=0.3,
        zorder=3,
        label="Positive effect"
    )

    ax.scatter(
        sig_effects[neg],
        y_jitter[neg],
        color=neg_color,
        s=point_size,
        edgecolor=neg_color,
        linewidth=0.3,
        zorder=3,
        label="Negative effect"
    )

    # ==================================================
    # Axes formatting
    # ==================================================
    ax.set_xlabel("Effect size", fontsize=30)
    ax.set_ylabel("Density", fontsize=30)
    ax.tick_params(axis="both", labelsize=20)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)
    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1, color="grey", alpha=0.7)
    ax.minorticks_on()
    ax.grid(
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=0.3,
        color="grey",
        alpha=0.5
    )

    # Start dots at y=0
    #ax.set_ylim(bottom=0.00000001)
    #ax.margins(y=0.01, x=0.01)

    #ax.legend(frameon=False, fontsize=6)

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
plot_effect_size_bellcurve_filled_with_sig_points(
    file_glob="../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
              "gwas_roh/roh_gwas_results_windows/1500_50/"
              "roh_gwas_windows_ISLAND_*.tsv",
    sig_df = sig_df_IL,
    effect_col="BETA_ROH_STATUS",
    neg_color="#B22222",   
    pos_color="#7B1FA2",
    fig_width_cm=15,
    fig_height_cm=15,
    point_size=75,
    out_file="ROH_GWAS_IL_effect_poster.svg"
    
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
    fig_width_cm=15,
    fig_height_cm=15,
    point_size=75,
    out_file="ROH_GWAS_ML_effect_poster.svg"
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

#### 11.2.1. Mainland


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1000_50/"
        "roh_gwas_windows_MAINLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (MAINLAND) – 1000 SNPs, 50% threshold",
    neff_genome=951
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1000_50/roh_gwas_windows_MAINLAND_*.tsv"),
    "ROH-GWAS QQ plot (MAINLAND) - 1000 snps, 50% threshold"
)

```


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "gwas_roh/roh_gwas_results_windows/1000_50/"
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
print(f"Total windows loaded: {df.shape[0]}")

neff_genome_ML = 951  # your simpleM result

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

```


```python

```


```python

```


```python

```


```python

```

#### 12.2.2. Island


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1000_50/"
        "roh_gwas_windows_ISLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (ISLAND) – 1000 SNPs, 50% threshold",
    neff_genome=1410
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1000_50/roh_gwas_windows_ISLAND_*.tsv"),
    "ROH-GWAS QQ plot (ISLAND) - 1000 snps, 50% threshold"
)

```


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/bf/"
    "gwas_roh/roh_gwas_results_windows/1000_50/"
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
print(f"Total windows loaded: {df.shape[0]}")

neff_genome_IL = 1410  # your simpleM result

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

```


```python

```


```python

```

#### 11.3.1. Mainland


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_80/"
        "roh_gwas_windows_MAINLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (MAINLAND) – 1500 SNPs, 80% threshold",
    neff_genome=947
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_80/roh_gwas_windows_MAINLAND_*.tsv"),
    "ROH-GWAS QQ plot (MAINLAND) - 1500 snps, 80% threshold"
)

```

#### 11.3.2. Island


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/1500_80/"
        "roh_gwas_windows_ISLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (ISLAND) – 1500 SNPs, 80% threshold",
    neff_genome=1409
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/1500_80/roh_gwas_windows_ISLAND_*.tsv"),
    "ROH-GWAS QQ plot (ISLAND) - 1500 snps, 80% threshold"
)

```

#### 11.4.1. Mainland


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/2500_50/"
        "roh_gwas_windows_MAINLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (MAINLAND) – 2500 SNPs, 50% threshold",
    neff_genome=947
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/2500_50/roh_gwas_windows_MAINLAND_*.tsv"),
    "ROH-GWAS QQ plot (MAINLAND) - 5500 snps, 50% threshold"
)

```

#### 11.4.2. Island


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/"
        "gwas_roh/roh_gwas_results_windows/2500_50/"
        "roh_gwas_windows_ISLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (ISLAND) – 2500 SNPs, 50% threshold",
    neff_genome=947
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_windows/2500_50/roh_gwas_windows_ISLAND_*.tsv"),
    "ROH-GWAS QQ plot (ISLAND) - 5500 snps, 50% threshold"
)

```


```python

```

##### SimpleM


```python
INPUT_DIR = "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/roh_gwas_input_windows/1500_50"

neff_ml, neff_ml_chr = compute_neff_from_window_folder(
    INPUT_DIR,
    pop="ML"
)

neff_il, neff_il_chr = compute_neff_from_window_folder(
    INPUT_DIR,
    pop="IL"
)

```

#### 11.5.1. Mainland

##### Exploratory plots


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_MAINLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (MAINLAND) – 1500 SNPs, 50% threshold SVL",
    neff_genome=1520
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/gwas_roh/roh_gwas_results_windows/1500_50/roh_gwas_windows_MAINLAND_*.tsv"),
    "ROH-GWAS QQ plot (MAINLAND) - 1500 snps, 50% threshold SVL"
)

```

##### Significant windows


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/svl/"
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

neff_genome_ML = 1520  # your simpleM result

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
plt.title("ROH-GWAS SVL effect sizes (Mainland)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

```


```python
INPUT_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/svl/"
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
    'PL':'#41AB5D', 'CV':'#BAE4B3', 'FV':'#BAE4B3'
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
INPUT_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/svl/"
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
    'PL':'#41AB5D', 'CV':'#BAE4B3', 'FV':'#BAE4B3'
}

windows_ML = df_plot_ML["WINDOW_ID"].unique()
print(f"Plotting {len(windows_ML)} significant windows")

for window in windows_ML:

    df_w_all = df_plot_ML[df_plot_ML["WINDOW_ID"] == window]

    # Get effect size from GWAS results
    beta = sig_df_ML.loc[
        sig_df_ML["WINDOW_ID"] == window,
        "BETA_ROH_STATUS"
    ].iloc[0]

    for sex in ["M", "F"]:

        df_w = df_w_all[df_w_all["Sex"] == sex]

        # Skip if too few data
        if df_w.empty:
            continue

        # Skip windows with no ROH variation within sex
        if df_w["ROH_STATUS"].nunique() < 2:
            continue

        plt.figure(figsize=(6, 4))

        # Boxplot
        sns.boxplot(
            x="ROH_STATUS",
            y="PHENO_z",
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
        plt.ylabel("Phenotype (z) SVL")
        plt.title(
            f"{window} | Sex: {sex}\nROH effect β = {beta:.3f}"
        )

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

#### 11.5.2. Island

##### Exploratory plots


```python
plot_manhattan(
    glob.glob(
        "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/"
        "gwas_roh/roh_gwas_results_windows/1500_50/"
        "roh_gwas_windows_ISLAND_*.tsv"
    ),
    "ROH-GWAS Manhattan plot (ISLAND) – 1500 SNPs, 50% threshold SVL",
    neff_genome=2535
)

plot_qq(
    glob.glob("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/gwas_roh/roh_gwas_results_windows/1500_50/roh_gwas_windows_ISLAND_*.tsv"),
    "ROH-GWAS QQ plot (ISLAND) - 1500 snps, 50% threshold SVL"
)

```

##### Significant windows


```python
# Path to your window-GWAS results
RESULTS_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/svl/"
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

neff_genome_IL = 2535  # your simpleM result

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
    "rPodSic1.hap1.1/roh_gwas/svl/"
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
    'PL':'#41AB5D', 'CV':'#BAE4B3', 'FV':'#BAE4B3'
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
INPUT_DIR = (
    "../results/population_analyses/"
    "rPodSic1.hap1.1/roh_gwas/svl/"
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
    'PL':'#41AB5D', 'CV':'#BAE4B3', 'FV':'#BAE4B3'
}

windows = df_plot_IL["WINDOW_ID"].unique()
print(f"Plotting {len(windows)} significant windows")

for window in windows:

    df_w_all = df_plot_IL[df_plot_IL["WINDOW_ID"] == window]

    # Get effect size from GWAS results
    beta = sig_df_IL.loc[
        sig_df_IL["WINDOW_ID"] == window,
        "BETA_ROH_STATUS"
    ].iloc[0]

    for sex in ["M", "F"]:

        df_w = df_w_all[df_w_all["Sex"] == sex]

        # Skip if no data for this sex
        if df_w.empty:
            continue

        # Skip windows with no ROH variation within sex
        if df_w["ROH_STATUS"].nunique() < 2:
            continue

        plt.figure(figsize=(6, 4))

        # Boxplot
        sns.boxplot(
            x="ROH_STATUS",
            y="PHENO_z",
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
        plt.title(
            f"{window} | Sex: {sex}\nROH effect β = {beta:.3f}"
        )

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


```python
sig_df_IL[["WINDOW_ID", "CHR", "BETA_ROH_STATUS", "N_ROH", "N_NONROH", "P_ROH_STATUS", "LOGP"]] \
    .to_csv("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/gwas_roh/roh_gwas_results_windows/1500_50/significant_windows/roh_neff_significant_windows_IL_svl.tsv", sep="\t", index=False)

```


```python
sig_df_ML[["WINDOW_ID", "CHR", "BETA_ROH_STATUS", "N_ROH", "N_NONROH", "P_ROH_STATUS", "LOGP"]] \
    .to_csv("../results/population_analyses/rPodSic1.hap1.1/roh_gwas/svl/gwas_roh/roh_gwas_results_windows/1500_50/significant_windows/roh_neff_significant_windows_ML_svl.tsv", sep="\t", index=False)

```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# --- Load GWAS results ---
gwas_file = "../results/population_analyses/rPodSic1.hap1.1/roh_gwas/bf/gwas_roh/roh_gwas_results_ML-IL/roh_gwas_OZ076857.1_ML_FULL.tsv"
df = pd.read_csv(gwas_file, sep="\t")

# --- Preprocess SNP info ---
df[['chrom','pos']] = df['SNP_ID'].str.split(':', expand=True)
df['pos'] = pd.to_numeric(df['pos'])

# Convert p-values and coefficients to numeric
for pred in ['SNProhA', 'SNProhB']:
    df[f'p_{pred}'] = pd.to_numeric(df[f'p_{pred}'], errors='coerce')
    df[f'{pred}'] = pd.to_numeric(df[f'{pred}'], errors='coerce')
    df[f'{pred}_log10p'] = -np.log10(df[f'p_{pred}'])

# Bonferroni threshold
n_snps = len(df)
bonf_thresh = 0.05 / n_snps
bonf_line = -np.log10(bonf_thresh)

# --- Manhattan plot for SNProhA ---
plt.figure(figsize=(15,5))
plt.scatter(df['pos'], df['SNProhA_log10p'], s=5, color='dodgerblue', alpha=0.7)
plt.axhline(bonf_line, color='red', linestyle='--', label='Bonferroni')
plt.xlabel('Position')
plt.ylabel('-log10(p-value)')
plt.title('GWAS Manhattan Plot: SNProhA ML')
plt.legend()
plt.show()

# --- Manhattan plot for SNProhB ---
plt.figure(figsize=(15,5))
plt.scatter(df['pos'], df['SNProhB_log10p'], s=5, color='orange', alpha=0.7)
plt.axhline(bonf_line, color='red', linestyle='--', label='Bonferroni')
plt.xlabel('Position')
plt.ylabel('-log10(p-value)')
plt.title('GWAS Manhattan Plot: SNProhB ML')
plt.legend()
plt.show()
```


```python

```


```python
beta_df = pd.DataFrame([
    # trait_group, trait, population, sex, beta_mean, beta_low, beta_high

    # ----------------------------
    # Physiological performance
    # ----------------------------
    ["Physiological performance", "Bite force", "Island", None,
     -3.5482, -6.1964, -1.1213],

    ["Physiological performance", "Bite force", "Mainland", None,
     -3.5482 + 7.9258, -6.1964 - 0.2403, -1.1213 + 16.9784],

    ["Physiological performance", "Sprint speed", "Island", None,
     -0.3993, -3.0506, 2.9016],

    ["Physiological performance", "Sprint speed", "Mainland", None,
     -0.3993 + 2.2160, -3.0506 - 7.3423, 2.9016 + 11.8007],

    # ----------------------------
    # Cognition
    # ----------------------------
    ["Cognition", "Spatial learning", "Island", None,
     -3.2790, -8.7185, 2.1795],

    ["Cognition", "Spatial learning", "Mainland", None,
     -3.2790 + 7.2065, -8.7185 - 13.9678, 2.1795 + 28.4195],

    # ----------------------------
    # Immune function
    # ----------------------------
    ["Immune function", "PHA", "Island", None,
     0.9820, -1.9394, 4.6328],

    ["Immune function", "PHA", "Mainland", None,
     0.9820 + 3.0403, -1.9394 - 7.7968, 4.6328 + 14.2091],

    # ----------------------------
    # Sperm quality
    # ----------------------------
    ["Sperm quality", "Sperm count", "Island", None,
     3.6640, -5.5056, 13.4636],

    ["Sperm quality", "Sperm count", "Mainland", None,
     3.6640 - 14.7551, -5.5056 - 57.0116, 13.4636 + 26.7705],

    ["Sperm quality", "Sperm length", "Island", None,
     -142.0420, -484.9891, 137.2284],

    ["Sperm quality", "Sperm length", "Mainland", None,
     -142.0420 + 510.7875, -484.9891 - 129.9171, 137.2284 + 1112.3841],

    ["Sperm quality", "Sperm velocity", "Island", None,
     -2.0387, -5.0270, 0.4513],

    ["Sperm quality", "Sperm velocity", "Mainland", None,
     -2.0387 + 5.5598, -5.0270 - 1.3515, 0.4513 + 12.6222],

    # ----------------------------
    # Morphology (sex-specific)
    # ----------------------------
    ["Morphology", "SVL", "Island", "Female",
     -1.1845, -2.1772, -0.0903],

    ["Morphology", "SVL", "Mainland", "Female",
     -1.1845 + 2.3807, -2.1772 + 1.0229, -0.0903 + 3.7563],

    ["Morphology", "SVL", "Island", "Male",
     -1.1845 + 0.1924, -2.1772 - 0.6873, -0.0903 + 1.0778],

    ["Morphology", "SVL", "Mainland", "Male",
     -1.1845 + 0.1924 + 2.3807,
     -2.1772 - 0.6873 + 1.0229,
     -0.0903 + 1.0778 + 3.7563],
],
columns=[
    "trait_group", "trait", "population", "sex",
    "beta_mean", "beta_low", "beta_high"
])

beta_df

```


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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# =========================
# Font (MATCHED)
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
# Colors
# =========================
palette = {"Mainland": "#208438", "Island": "#479BC9"}


def plot_beta_combined(
    df,
    figsize_cm=(9.53, 9.15),   # <-- YOU CONTROL SIZE HERE
    x_label="Inbreeding depression (β)",  # <-- YOU CONTROL LABEL HERE
    xlim=(-26, 26)
):

    # =========================
    # Order traits
    # =========================
    order = df["trait"].unique()[::-1]
    y_positions = {trait: i for i, trait in enumerate(order)}

    offset = {"Island": -0.15, "Mainland": +0.15}

    # =========================
    # Figure
    # =========================
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(figsize_cm[0]*cm, figsize_cm[1]*cm),
        dpi=300
    )

    # =========================
    # Plot
    # =========================
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
            alpha=0.9,
            zorder=2
        )

    # =========================
    # Reference line
    # =========================
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)

    # =========================
    # Grid
    # =========================
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.grid(axis="y", visible=False)

    # =========================
    # Y axis (traits)
    # =========================
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)

    # =========================
    # X axis
    # =========================
    ax.set_xlim(xlim)

    # =========================
    # Labels (USER CONTROLLED)
    # =========================
    ax.set_xlabel(x_label)
    ax.set_ylabel("")

    # =========================
    # Ticks (EXPLICIT CONTROL)
    # =========================
    ax.tick_params(
        axis='x',
        which='major',
        bottom=True,
        top=False,
        length=3,
        width=0.8,
        direction='out',
        pad=2
    )

    ax.tick_params(
        axis='y',
        which='major',
        left=True,
        right=False,
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
    # Layout (MATCH YOUR SYSTEM)
    # =========================
    ax.set_position([0.25, 0.12, 0.7, 0.82])

    # =========================
    # Legend
    # =========================
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=palette[k], markersize=5, label=k)
        for k in palette
    ]

    ax.legend(handles=handles, frameon=False, loc="lower right")

    return fig, ax
```


```python
fig, ax = plot_beta_combined(
    beta_df_std,
    xlim=(-30, 30)                       # optional
)

plt.savefig("beta_combined.svg", dpi=300)
plt.show()
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

sns.set_style("whitegrid")

COLORS = {
    "Mainland": "#009900",  # green
    "Island": "#0066CC"     # blue
}

# # --- Register Arial ---
# arial_path = "/scratch/antwerpen/206/vsc20666/arial.ttf"
# fm.fontManager.addfont(arial_path)
# arial_font = fm.FontProperties(fname=arial_path)

# plt.rcParams.update({
#     "font.family": arial_font.get_name(),
#     "font.size": 8,
#     "axes.labelsize": 8,
#     "axes.titlesize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8
# })


def plot_beta_trait(df, trait_name):
    d = df[df["trait"] == trait_name].reset_index(drop=True)

    width_cm = 7
    height_cm = 3
    fig, ax = plt.subplots(
        figsize=(width_cm / 2.54, height_cm / 2.54)
    )

    if len(d) == 2:
        y = np.array([-0.25, 0.25])
    else:
        y = np.linspace(-0.3, 0.3, len(d))

    for i, row in d.iterrows():
        ax.errorbar(
            row["beta_mean"],
            y[i],
            xerr=[[row["beta_mean"] - row["beta_low"]],
                  [row["beta_high"] - row["beta_mean"]]],
            fmt="o",
            color=COLORS[row["population"]],
            capsize=3,
            markersize=4,
            linewidth=1.2,
            alpha=0.9
        )

    # --- Fixed x-axis across all traits ---
    ax.set_xlim(-26, 26)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_yticks([])
    ax.set_yticklabels("")
    ax.set_ylim(-0.6, 0.6)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    if ax.get_legend() is not None:
        ax.legend_.remove()

    ax.grid(
        axis="x",
        color="grey",
        linestyle="--",
        linewidth=0.6,
        alpha=0.5
    )
    ax.grid(axis="y", visible=False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", colors="black", width=0.8, length=3)

    plt.tight_layout(pad=0.6)

    return fig, ax

```


```python
plot_beta_trait(beta_df, "Bite force")
plot_beta_trait(beta_df, "Sprint speed")
plot_beta_trait(beta_df, "PHA")
plot_beta_trait(beta_df, "Spatial learning")
```


```python
plot_beta_trait(beta_df_std, "Bite force")
plot_beta_trait(beta_df_std, "Sprint speed")
plot_beta_trait(beta_df_std, "PHA")
plot_beta_trait(beta_df_std, "Spatial learning")
plot_beta_trait(beta_df_std, "Sperm count")
plot_beta_trait(beta_df_std, "Sperm length")
plot_beta_trait(beta_df_std, "Sperm velocity")
```


```python
fig, ax = plot_beta_trait(beta_df_std, "Bite force")
fig.savefig("bite_force_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df_std, "Sprint speed")
fig.savefig("sprint_speed_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df_std, "PHA")
fig.savefig("PHA_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df_std, "Spatial learning")
fig.savefig("Spatial_learning_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df_std, "Sperm count")
fig.savefig("Sperm_count_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df_std, "Sperm length")
fig.savefig("Sperm_length_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df_std, "Sperm velocity")
fig.savefig("Sperm_velocity_inbreeding.svg", format="svg")
plt.close(fig)
```


```python

```


```python

```


```python
def plot_beta_trait(df, trait_name):
    d = df[df["trait"] == trait_name].reset_index(drop=True)

    # --- Figure size: 7 cm wide ---
    width_cm = 7
    height_cm = 2.50
    fig, ax = plt.subplots(
        figsize=(width_cm / 2.54, height_cm / 2.54)
    )

    # --- Centered y positions ---
    if len(d) == 2:
        y = np.array([-0.25, 0.25])
    else:
        y = np.linspace(-0.3, 0.3, len(d))

    for i, row in d.iterrows():
        ax.errorbar(
            row["beta_mean"],
            y[i],
            xerr=[[row["beta_mean"] - row["beta_low"]],
                  [row["beta_high"] - row["beta_mean"]]],
            fmt="o",
            color=COLORS[row["population"]],
            capsize=3,
            markersize=4,
            linewidth=1.2,
            alpha=0.9
        )

    # --- Center zero on x-axis ---
    max_abs = max(abs(d["beta_low"].min()), abs(d["beta_high"].max()))
    ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # --- Y-axis ---
    ax.set_yticks(y)
    ax.set_yticklabels("")
    ax.set_ylim(-0.6, 0.6)

    # --- Remove labels, title, legend ---
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    if ax.get_legend() is not None:
        ax.legend_.remove()

    # --- Grey vertical grid lines ---
    ax.grid(
        axis="x",
        color="grey",
        linestyle="--",
        linewidth=0.6,
        alpha=0.5
    )
    ax.grid(axis="y", visible=False)

    # --- Black axes ---
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", colors="black", width=0.8, length=3)

    plt.tight_layout(pad=0.6)

    return fig, ax

```


```python
plot_beta_trait(beta_df, "Sperm count")
plot_beta_trait(beta_df, "Sperm length")
plot_beta_trait(beta_df, "Sperm velocity")
```


```python
fig, ax = plot_beta_trait(beta_df, "Sperm count")
fig.savefig("Sperm_count_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df, "Sperm length")
fig.savefig("Sperm_length_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
fig, ax = plot_beta_trait(beta_df, "Sperm velocity")
fig.savefig("Sperm_velocity_inbreeding.svg", format="svg")
plt.close(fig)
```


```python
def plot_beta_trait_svl(df):
    # --- Filter and FORCE correct order ---
    d = (
        df[df["trait"] == "SVL"]
        .assign(
            pop_order=lambda x: x["population"].map(
                {"Mainland": 0, "Island": 1}
            ),
            sex_order=lambda x: x["sex"].map(
                {"Female": 0, "Male": 1}
            )
        )
        .sort_values(["pop_order", "sex_order"])
        .reset_index(drop=True)
    )

    # --- Figure size: 7 cm wide ---
    width_cm = 7
    height_cm = 4
    fig, ax = plt.subplots(
        figsize=(width_cm / 2.54, height_cm / 2.54)
    )

    # --- Centered y positions (top to bottom) ---
    y_positions = np.array([0.30, 0.10, -0.10, -0.30])

    marker_map = {
        "Male": "o",     # circles
        "Female": "^"    # triangles
    }

    for i, row in d.iterrows():
        ax.errorbar(
            row["beta_mean"],
            y_positions[i],
            xerr=[[row["beta_mean"] - row["beta_low"]],
                  [row["beta_high"] - row["beta_mean"]]],
            fmt=marker_map[row["sex"]],
            color=COLORS[row["population"]],
            capsize=3,
            markersize=4,
            linewidth=1.2,
            alpha=0.9
        )

    # --- Center zero on x-axis ---
    max_abs = max(abs(d["beta_low"].min()), abs(d["beta_high"].max()))
    ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # --- Y-axis ---
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)

    # --- Remove labels, title, legend ---
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    if ax.get_legend() is not None:
        ax.legend_.remove()

    # --- Grey vertical grid lines ---
    ax.grid(
        axis="x",
        color="grey",
        linestyle="--",
        linewidth=0.6,
        alpha=0.5
    )
    ax.grid(axis="y", visible=False)

    # --- Black axes ---
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", colors="black", width=0.8, length=3)

    plt.tight_layout(pad=0.6)

    return fig, ax

```


```python
plot_beta_trait_svl(beta_df)
```


```python
fig, ax = plot_beta_trait_svl(beta_df)
fig.savefig("svl_inbreeding.svg", format="svg")
plt.close(fig)
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
df.head()
```


```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54))

# ---- Raw data (background) ----
for t, color in zip(["island", "mainland"], ["#B2182B", "#2166AC"]):
    d_raw = df[df["type"] == t]

    ax.scatter(
        d_raw["Froh"],
        d_raw["log_BF_max"],
        s=12,
        alpha=0.75,
        color=color,
        edgecolor="none"
    )

# ---- Predictions (foreground) ----
pred_sub = pred[(pred["Froh"] >= 0.05) & (pred["Froh"] <= 0.65)]

for t, color in zip(["island", "mainland"], ["#B2182B", "#2166AC"]):
    d = pred_sub[pred_sub["type"] == t]

    ax.fill_between(
        d["Froh"],
        np.log(d["BF_low"]),
        np.log(d["BF_high"]),
        alpha=0.25,
        color=color
    )
    ax.plot(
        d["Froh"],
        np.log(d["BF_mean"]),
        color=color,
        linewidth=2.0,
        label=t
    )

# ---- Labels & styling ----
ax.set_xlabel("Genomic inbreeding (Froh)")
ax.set_ylabel("Predicted log(bite force)")

ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))

for spine in ax.spines.values():
    spine.set_color("black")

ax.set_xlim(0.05, 0.65)

plt.tight_layout()
plt.show()

```


```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54))

# =========================
# Raw data (Island only)
# =========================
d_raw = df[df["type"] == "island"]

ax.scatter(
    d_raw["Froh"],
    d_raw["log_BF_max"],
    s=12,
    alpha=0.75,
    color="#B2182B",
    edgecolor="none"
)

# =========================
# Predictions (Island only)
# =========================
pred_sub = pred[(pred["Froh"] >= 0.05) & (pred["Froh"] <= 0.65)]
d = pred_sub[pred_sub["type"] == "island"]

ax.fill_between(
    d["Froh"],
    np.log(d["BF_low"]),
    np.log(d["BF_high"]),
    alpha=0.25,
    color="#B2182B"
)

ax.plot(
    d["Froh"],
    np.log(d["BF_mean"]),
    color="#B2182B",
    linewidth=2.0
)

# =========================
# Labels & styling
# =========================
ax.set_xlabel("Genomic inbreeding (Froh)")
ax.set_ylabel("Predicted log(bite force)")

# (legend removed since only one group)

for spine in ax.spines.values():
    spine.set_color("black")

ax.set_xlim(0.5, 0.65)

plt.tight_layout()
plt.show()
```


```python

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
# Colors (MATCH YOUR PALETTE)
# =========================
color = "#479BC9"  # Island blue (same as other panels)

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
pred_sub = pred[(pred["Froh"] >= 0.50) & (pred["Froh"] <= 0.70)]
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
# Grid (MATCHED)
# =========================
ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
ax.grid(axis="y", visible=False)

# =========================
# Axes
# =========================
ax.set_xlim(0.50, 0.65)

ax.set_xlabel("Genomic inbreeding (FROH)", labelpad=6)
ax.set_ylabel("log(Bite force)", labelpad=6)

# =========================
# Tick styling (MATCHED)
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
# Spines (MATCHED BLACK BOX)
# =========================
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.8)

# =========================
# Layout (CRUCIAL MATCH)
# =========================
ax.set_position([0.18, 0.18, 0.76, 0.74])

# =========================
# Finalize
# =========================
fig.canvas.draw()

plt.show()
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


```python

```


```python
import matplotlib as mpl

mpl.rcParams["font.family"] = "DejaVu Sans"
```


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_beta_trait(
    df,
    trait_name,
    y_label="",
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10
):
    d = df[df["trait"] == trait_name].reset_index(drop=True)

    # --- Figure size ---
    width_cm = 25
    height_cm = 8
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))

    # --- Centered y positions ---
    if len(d) == 2:
        y = np.array([-0.25, 0.25])
    else:
        y = np.linspace(-0.3, 0.3, len(d))

    # --- Plot points + CI ---
    for i, row in d.iterrows():
        ax.errorbar(
            row["beta_mean"],
            y[i],
            xerr=[[row["beta_mean"] - row["beta_low"]],
                  [row["beta_high"] - row["beta_mean"]]],
            fmt="o",
            color=COLORS[row["population"]],
            capsize=10,
            markeredgewidth=5,
            markersize=25,
            linewidth=5,
            alpha=1
        )

    # --- Center zero on x-axis ---
    max_abs = max(abs(d["beta_low"].min()), abs(d["beta_high"].max()))
    ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder = 0)

    # --- Y-axis ---
    ax.set_yticks(y)
    ax.set_yticklabels("")
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel(y_label, fontsize=label_fontsize)

    # --- X ticks ---
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_xlabel("Inbreeding depression (β)", fontsize=35)

    # --- Grid ---
    ax.grid(
        axis="x",
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )
    ax.grid(axis="y", visible=False)

    # --- Legend order ---
    legend_order = ["Mainland", "Island"]
    
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=pop,
            markerfacecolor=COLORS[pop],
            markersize=12
        )
        for pop in legend_order
        if pop in d["population"].values
    ]
    
    ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=30,
        loc="lower right",
        handletextpad=0.001,
        labelspacing=0.001
    )


    plt.tight_layout(pad=0.6)

    return fig, ax

```


```python
plot_beta_trait(beta_df, "Bite force")
```


```python
fig, ax = plot_beta_trait(beta_df, "Bite force")
fig.savefig("bite_force_inbreeding_poster.svg", format="svg")
plt.close(fig)


```


```python
print(plt.rcParams["font.family"])
print(plt.rcParams["font.size"])
```


```python
import matplotlib as mpl

mpl.rcParams["font.family"] = "DejaVu Sans"
```


```python
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "island": "#0066CC",
    "mainland": "#009900"
}

fig, ax = plt.subplots(figsize=(25/2.54, 10/2.54))

# -------------------------------
# Raw data (background)
# -------------------------------
for t in ["mainland", "island"]:   # mainland plotted first (under)
    d_raw = df[df["type"] == t]

    ax.scatter(
        d_raw["Froh"],
        d_raw["log_BF_max"],
        s=250,
        alpha=0.25,                 # subtle background
        color=COLORS[t],
        edgecolor="none",
        zorder=1
    )

# -------------------------------
# Predictions (foreground)
# -------------------------------
pred_sub = pred[(pred["Froh"] >= 0.05) & (pred["Froh"] <= 0.65)]

for t in ["mainland", "island"]:
    d = pred_sub[pred_sub["type"] == t]

    ax.fill_between(
        d["Froh"],
        np.log(d["BF_low"]),
        np.log(d["BF_high"]),
        color=COLORS[t],
        alpha=0.25,
        zorder=2
    )

    ax.plot(
        d["Froh"],
        np.log(d["BF_mean"]),
        color=COLORS[t],
        linewidth=5,
        label=t.capitalize(),
        zorder=3
    )

# -------------------------------
# Axes, labels, grid
# -------------------------------
ax.set_xlabel("Froh", fontsize=35)
ax.set_ylabel("log(BF)", fontsize=35)

ax.set_xlim(0.05, 0.65)

ax.grid(
    axis="x",
    color="grey",
    linestyle="--",
    linewidth=1,
    alpha=0.5
)
ax.grid(axis="y", visible=False)

ax.tick_params(axis="both", labelsize=30, width=0.8, length=3)

# -------------------------------
# Legend (style-matched)
# -------------------------------
ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=30,
    handletextpad=0.3,
    labelspacing=0.3
)

plt.tight_layout(pad=0.6)
plt.savefig("model_inbreeding_depression_poster.svg", format="svg", dpi=300)
plt.show()

```


```python

```


```python

```


```python
df_load = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/genetic_load_counts.tsv", sep="\t")
df_load.head()
```


```python
df_load.groupby(["population_type","category"])[["Hom","Het","Tot"]].mean()
```


```python
df_plot = df_load.melt(
    id_vars=["sample","population","population_type","category"],
    value_vars=["Tot","Hom","Het"],
    var_name="genotype",
    value_name="count"
)

order = ["Tot","Hom","Het"]
```


```python
color_map = { 'PJ':'#6BAED6', 'MP': '#2171B5', 'KP': '#4292C6', 'PM': '#08306B', 'BJ': '#08519C', 'SC': '#9ECAE1', 'PK': '#C6DBEF', 'VS': '#238B45', 'TM': '#00441B', 'RG': '#BAE4B3', 'SP': '#74C476', 'TR': '#006D2C', 'PL': '#41AB5D' } 
desired_order = ["PL", "TR", "SP", "TM", "RG", "VS", "MP", "SC", "BJ", "KP", "PK", "PM", "PJ"]
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.catplot(
    data=df_plot,
    x="genotype",
    order=order,
    y="count",
    hue="population",
    hue_order=desired_order,   # enforce population order
    palette=color_map,         # apply custom colors
    row="category",
    kind="strip",
    dodge=True,
    jitter=True,
    height=3,
    aspect=2,
    sharey=False
)

g.set_axis_labels("Genotype", "Count")
g.tight_layout()

plt.show()
```


```python

```


```python
import seaborn as sns
import matplotlib.pyplot as plt

colors = {"mainland": "green", "island": "blue"}

g = sns.catplot(
    data=df_plot,
    x="genotype",
    order=order,
    y="count",
    hue="population_type",
    hue_order=["mainland","island"],
    row="category",
    row_order=["SYN","DEL","LOF"],   # top → bottom order
    kind="strip",
    dodge=True,
    jitter=True,
    height=3,
    aspect=2,
    sharey=False,
    palette=colors
)

g.set_axis_labels("Genotype", "Count")
g.set_titles("{row_name}")

g._legend.remove()

plt.tight_layout()
plt.show()
```


```python

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


```python
import seaborn as sns
import matplotlib.pyplot as plt
colors = {"mainland": "green", "island": "blue"}
g = sns.catplot(
    data=df_ratio_plot,
    x="population_type",
    y="ratio",
    hue="population_type",
    row="metric",
    kind="strip",
    jitter=True,
    height=3.5,
    aspect=2,
    sharey=False,
    palette=colors
)

g.set_axis_labels("Population", "Ratio")
g.set_titles("{row_name}")
g.tight_layout()

plt.show()
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.catplot(
    data=df_ratio_plot,
    x="population",
    y="ratio",
    hue="population",
    hue_order=desired_order,
    palette=color_map,
    row="metric",
    kind="strip",
    jitter=True,
    height=3.5,
    aspect=2,
    sharey=False
)

g.set_axis_labels("Population", "Ratio")
g.set_titles("{row_name}")
g.tight_layout()

plt.show()
```


```python

```


```python

```


```python
import seaborn as sns
import matplotlib.pyplot as plt
colors = {"mainland": "green", "island": "blue"}
g = sns.catplot(
    data=df_ratio_plot,
    x="population_type",
    y="ratio",
    hue="population_type",
    row="metric",
    kind="strip",
    jitter=True,
    height=3.5,
    aspect=2,
    sharey=False,
    palette=colors
)

g.set_axis_labels("Population", "Ratio")
g.set_titles("{row_name}")
g.tight_layout()

plt.show()
```


```python
df_ratio["HomDEL_SYN"] = df_ratio["Hom_DEL"] / df_ratio["Hom_SYN"]
df_ratio["HetDEL_SYN"] = df_ratio["Het_DEL"] / df_ratio["Het_SYN"]

df_ratio["HomLOF_SYN"] = df_ratio["Hom_LOF"] / df_ratio["Hom_SYN"]
df_ratio["HetLOF_SYN"] = df_ratio["Het_LOF"] / df_ratio["Het_SYN"]
```


```python
df_ratio_plot = df_ratio.melt(
    id_vars=["sample","population","population_type"],
    value_vars=[
        "HomDEL_SYN",
        "HetDEL_SYN",
        "HomLOF_SYN",
        "HetLOF_SYN"
    ],
    var_name="metric",
    value_name="ratio"
)
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.catplot(
    data=df_ratio_plot,
    x="population_type",
    y="ratio",
    hue="population_type",
    row="metric",
    kind="strip",
    jitter=True,
    height=3.5,
    aspect=2,
    sharey=False
)

g.set_axis_labels("Population type", "Ratio")
g.set_titles("{row_name}")
g.tight_layout() 

plt.show()
```


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
g = sns.catplot(
    data=ns,
    x="bin",
    y="N_S",
    col="group",
    kind="bar",
    col_wrap=4,
    height=4,
    aspect=1.2,
    sharey=False
)

g.set_axis_labels("Allele frequency bin", "N/S ratio")
g.set_titles("{col_name}")

plt.tight_layout()
plt.show()
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

# keep only the two groups
ns_subset = ns.query("group in ['Mainland', 'Island']")

# enforce bin order (optional but recommended)
bin_order = ["0-5%", "5-10%", "10-20%", "20-80%", ">80%"]

g = sns.catplot(
    data=ns_subset,
    x="bin",
    y="N_S",
    col="group",
    col_order=["Mainland", "Island"],   # only these two panels
    kind="bar",
    order=bin_order,
    height=4,
    aspect=1.2,
    sharey=True                         # same y-axis scale
)

# color panels
colors = {"Mainland": "green", "Island": "blue"}

for ax in g.axes.flat:
    title = ax.get_title().split(" = ")[-1]
    for bar in ax.patches:
        bar.set_facecolor(colors[title])

g.set_axis_labels("Allele frequency bin", "N/S ratio")
g.set_titles("{col_name}")

plt.tight_layout()
plt.show()
```


```python
bin_order = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

g = sns.catplot(
    data=sfs,
    x="bin",
    y="prop",
    hue="category",
    col="group",
    kind="bar",
    col_wrap=4,
    height=4,
    aspect=1.2,
    sharey=False,
    order=bin_order
)

g.set_axis_labels("Allele frequency bin", "Proportion of variants")
g.set_titles("{col_name}")

plt.tight_layout()
plt.show()
```


```python
hue_order = ["SYN", "DEL", "LOF"]


sfs_subset["category"] = pd.Categorical(
    sfs_subset["category"],
    categories=hue_order,
    ordered=True
)

g = sns.catplot(
    data=sfs_subset,
    x="bin",
    y="prop",
    hue="category",
    hue_order=hue_order,
    col="group",
    col_order=["Mainland","Island"],
    kind="bar",
    order=bin_order,
    height=4,
    aspect=1.2,
    sharey=True
)
```


```python
sfs.head()
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Data prep
# -----------------------
bin_order_ns = ["0-5%", "5-10%", "10-20%", "20-80%", ">80%"]
bin_order_sfs = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

ns_subset = ns.query("group in ['Mainland', 'Island']")
sfs_subset = sfs.query("group in ['Mainland', 'Island']")

hue_order = ["SYN", "DEL", "LOF"]
sfs_subset["category"] = pd.Categorical(
    sfs_subset["category"],
    categories=hue_order,
    ordered=True
)

# -----------------------
# Figure layout
# -----------------------
fig, axes = plt.subplots(
    2, 2, 
    figsize=(10, 8),
    sharex=False,     # ✅ separate x-axes
    sharey='row'
)

groups = ["Mainland", "Island"]
colors = {"Mainland": "green", "Island": "blue"}

# -----------------------
# TOP ROW: N/S ratio
# -----------------------
for i, group in enumerate(groups):
    ax = axes[0, i]
    data = ns_subset[ns_subset["group"] == group]

    sns.barplot(
        data=data,
        x="bin",
        y="N_S",
        order=bin_order_ns,
        ax=ax,
        color=colors[group]
    )

    ax.set_title(group)
    ax.set_xlabel("")
    ax.set_ylabel("N/S ratio")

# -----------------------
# BOTTOM ROW: SFS
# -----------------------
palette = {
    "SYN": "#4daf4a",
    "DEL": "#e41a1c",
    "LOF": "#377eb8"
}

for i, group in enumerate(groups):
    ax = axes[1, i]
    data = sfs_subset[sfs_subset["group"] == group]

    sns.barplot(
        data=data,
        x="bin",
        y="prop",
        hue="category",
        hue_order=hue_order,
        order=bin_order_sfs,
        palette=palette,
        ax=ax
    )

    ax.set_xlabel("Allele frequency bin")
    ax.set_ylabel("Proportion")

    if i == 1:
        ax.legend(title="Category")
    else:
        ax.get_legend().remove()

# -----------------------
# Final styling
# -----------------------
plt.tight_layout()
plt.show()
```


```python

```




```python

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
                linewidth=0,
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
                linewidth=0,
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
        ax.tick_params(axis='x', bottom=True, labelbottom=True, length=3, width=0.8)

        # =========================
        # Y-axis → ONLY LEFT COLUMN
        # =========================
        if j == 0:
            ax.tick_params(axis='y', left=True, labelleft=True, length=3, width=0.8)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

        # Clean labels
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Titles
        if i == 0:
            ax.set_title(group, fontsize=8)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.90,
    bottom=0.12,
    wspace=0.05,
    hspace=0.30
)

# =========================
# Save
# =========================
plt.savefig("Figure_NS_SFS_final.svg", format="svg", dpi=300)
plt.show()
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


```python

```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/rohs/roh_load_summary.tsv", sep="\t")

# =========================
# Compute load ratios
# =========================
df["del_total"] = df["total_del"] / df["total_syn"]
df["lof_total"] = df["total_lof"] / df["total_syn"]

df["del_hom"] = df["hom_del"] / df["hom_syn"]
df["lof_hom"] = df["hom_lof"] / df["hom_syn"]

df["del_het"] = df["het_del"] / df["het_syn"]
df["lof_het"] = df["het_lof"] / df["het_syn"]

# =========================
# Setup
# =========================
regions = ["mainland", "island"]
locations = ["in_roh", "out_roh"]

metrics = {
    "DEL/SYN (total)": "del_total",
    "DEL/SYN (hom)": "del_hom",
    "DEL/SYN (het)": "del_het",
    "LOF/SYN (total)": "lof_total",
    "LOF/SYN (hom)": "lof_hom",
    "LOF/SYN (het)": "lof_het",
}

# =========================
# Plot
# =========================
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

positions = [0, 1, 3, 4]  # spacing between mainland and island

for ax, (title, metric) in zip(axes, metrics.items()):

    data = []
    x_positions = []

    for i, region in enumerate(regions):
        for j, loc in enumerate(locations):

            subset = df[(df["region"] == region) & (df["location"] == loc)][metric].dropna()

            pos = i * 3 + j  # creates gap between mainland and island
            data.append(subset)
            x_positions.append(pos)

            # raw points
            jitter = np.random.normal(0, 0.04, size=len(subset))
            ax.scatter(np.full(len(subset), pos) + jitter, subset,
                       s=12, alpha=0.7)

    # violin plot
    vp = ax.violinplot(data, positions=x_positions, widths=0.6,
                       showmeans=False, showmedians=True, showextrema=False)

    # style violins
    for body in vp['bodies']:
        body.set_alpha(0.3)

    # x-axis formatting
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["mainland", "island"])

    ax.set_title(title, fontsize=10)

    # clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Y labels
axes[0].set_ylabel("Genetic load")
axes[3].set_ylabel("Genetic load")

plt.tight_layout()
plt.show()
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/rohs/roh_load_summary.tsv", sep="\t")

# =========================
# Compute load ratios
# =========================
df["del_total"] = df["total_del"] / df["total_syn"]
df["lof_total"] = df["total_lof"] / df["total_syn"]

# =========================
# Setup
# =========================
regions = ["mainland", "island"]
locations = ["in_roh", "out_roh"]

metrics = {
    "DEL/SYN": "del_total",
    "LOF/SYN": "lof_total",
}

# consistent colors
colors = {
    "in_roh": "#4C72B0",   # blue
    "out_roh": "#DD8452"   # orange
}

# =========================
# Plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, (title, metric) in zip(axes, metrics.items()):

    data = []
    x_positions = []
    color_list = []

    for i, region in enumerate(regions):
        for j, loc in enumerate(locations):

            subset = df[(df["region"] == region) & (df["location"] == loc)][metric].dropna()

            pos = i * 3 + j
            data.append(subset)
            x_positions.append(pos)
            color_list.append(colors[loc])

            # raw points
            jitter = np.random.normal(0, 0.04, size=len(subset))
            ax.scatter(np.full(len(subset), pos) + jitter, subset,
                       s=14, alpha=0.7, color=colors[loc])

    # violin plot
    vp = ax.violinplot(data, positions=x_positions, widths=0.6,
                       showmeans=False, showmedians=True, showextrema=False)

    # color violins consistently by ROH status
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(color_list[i])
        body.set_alpha(0.3)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    # x-axis
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["mainland", "island"])

    ax.set_title(title, fontsize=11)

    # clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# y label
axes[0].set_ylabel("Genetic load")

# legend (manual, clean)
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='in ROH',
               markerfacecolor=colors["in_roh"], markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='out ROH',
               markerfacecolor=colors["out_roh"], markersize=6)
]
axes[1].legend(handles=handles, frameon=False)

plt.tight_layout()
plt.show()
```


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

plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

# =========================
# Load data
# =========================
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/rohs/roh_load_summary.tsv", sep="\t")

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
# Colors (region-based)
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
    nrows=1,
    ncols=2,
    figsize=(9.53*cm, 4.5*cm),
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
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

    # Plot manually (to fully control colors)
    for j, region in enumerate(order):
        for k, loc in enumerate(locations):

            vals = sub[(sub["region"] == region) & (sub["location"] == loc)]["ratio"]

            if len(vals) == 0:
                continue

            # position (dodge)
            offset = -0.2 if loc == "in_roh" else 0.2
            x = j + offset

            color = palette[(region, loc)]

            # scatter (raw points)
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=14,
                alpha=0.6,
                color=color,
                zorder=1
            )

            # median line
            median = np.median(vals)
            ax.hlines(
                y=median,
                xmin=x - 0.15,
                xmax=x + 0.15,
                color=color,
                linewidth=2,
                zorder=2
            )

    # =========================
    # Style
    # =========================
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)

    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel("Genetic load" if i == 0 else "")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# =========================
# Legend
# =========================
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Mainland (in ROH)',
               markerfacecolor="#208438", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Mainland (out ROH)',
               markerfacecolor="#145022", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Island (in ROH)',
               markerfacecolor="#479BC9", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Island (out ROH)',
               markerfacecolor="#255E7D", markersize=5),
]

axes[1].legend(handles=handles, frameon=False, fontsize=5)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.88,
    bottom=0.15,
    wspace=0.35
)

# =========================
# Save
# =========================
plt.savefig("Figure_total_load_ROH.svg", format="svg", dpi=300)
plt.show()
```


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

plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

# =========================
# Load data
# =========================
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/rohs/roh_load_summary.tsv", sep="\t")

# =========================
# Reshape (RAW COUNTS)
# =========================
df_plot = pd.melt(
    df,
    id_vars=["sample", "region", "location"],
    value_vars=["total_del", "total_lof"],
    var_name="metric",
    value_name="value"
)

df_plot["metric"] = df_plot["metric"].map({
    "total_del": "Deleterious (count)",
    "total_lof": "LoF (count)"
})

# =========================
# Colors (region-based)
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
    nrows=1,
    ncols=2,
    figsize=(9.53*cm, 4.5*cm),
    dpi=300,
    sharex=True
)

metrics = ["Deleterious (count)", "LoF (count)"]

# =========================
# Plot
# =========================
for i, metric in enumerate(metrics):

    ax = axes[i]
    sub = df_plot[df_plot["metric"] == metric]

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)

    # Plot manually
    for j, region in enumerate(order):
        for k, loc in enumerate(locations):

            vals = sub[(sub["region"] == region) & (sub["location"] == loc)]["value"]

            if len(vals) == 0:
                continue

            # dodge
            offset = -0.2 if loc == "in_roh" else 0.2
            x = j + offset

            color = palette[(region, loc)]

            # scatter
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=14,
                alpha=0.6,
                color=color,
                zorder=1
            )

            # median
            median = np.median(vals)
            ax.hlines(
                y=median,
                xmin=x - 0.15,
                xmax=x + 0.15,
                color=color,
                linewidth=2,
                zorder=2
            )

    # =========================
    # Style
    # =========================
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)

    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel("Variant count" if i == 0 else "")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# =========================
# Legend
# =========================
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Mainland (in ROH)',
               markerfacecolor="#208438", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Mainland (out ROH)',
               markerfacecolor="#145022", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Island (in ROH)',
               markerfacecolor="#479BC9", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Island (out ROH)',
               markerfacecolor="#255E7D", markersize=5),
]

axes[1].legend(handles=handles, frameon=False, fontsize=5)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.88,
    bottom=0.15,
    wspace=0.35
)

# =========================
# Save
# =========================
plt.savefig("Figure_total_counts_ROH.svg", format="svg", dpi=300)
plt.show()
```


```python

```


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

plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

sns.set_style("whitegrid")

# =========================
# Load data
# =========================
df = pd.read_csv("../results/population_analyses/rPodSic1.hap1.1/genetic_load/rohs/roh_load_summary.tsv", sep="\t")

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
    figsize=(9.53*cm, 4.5*cm),
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

    # Grid (match style)
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.set_axisbelow(True)

    # FULL BOX around plot (important)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # -------------------------
    # Scatter + medians
    # -------------------------
    for j, region in enumerate(order):
        for loc in locations:

            vals = sub[(sub["region"] == region) & (sub["location"] == loc)]["ratio"]

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
                alpha=0.6,
                color=color,
                zorder=1
            )

            # median line
            median = np.median(vals)
            ax.hlines(
                y=median,
                xmin=x - 0.15,
                xmax=x + 0.15,
                color=color,
                linewidth=2,
                zorder=2
            )

    # =========================
    # Axes formatting (MATCHED)
    # =========================
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([""] * len(order))  # no labels

    # X ticks visible but no labels
    ax.tick_params(axis='x', bottom=True, labelbottom=False, length=3, width=0.8)

    # Y only left panel
    if i == 0:
        ax.tick_params(axis='y', left=True, labelleft=True, length=3, width=0.8)
        ax.set_ylabel("Genetic load")
    else:
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.set_ylabel("")

    ax.set_xlabel("")
    ax.set_title(metric, fontsize=8)

# =========================
# Legend (clean)
# =========================
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Mainland (in ROH)',
               markerfacecolor="#208438", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Mainland (out ROH)',
               markerfacecolor="#145022", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Island (in ROH)',
               markerfacecolor="#479BC9", markersize=5),
    plt.Line2D([0], [0], marker='o', color='w', label='Island (out ROH)',
               markerfacecolor="#255E7D", markersize=5),
]

axes[1].legend(handles=handles, frameon=False, fontsize=6)

# =========================
# Layout (MATCHED)
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.90,
    bottom=0.12,
    wspace=0.12
)

# =========================
# Save
# =========================
plt.savefig("Figure_total_load_ROH_final.svg", format="svg", dpi=300)
plt.show()
```


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
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
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
    
        # 🔥 KEY CHANGE: region-specific max
        local_vals = pd.concat([vals1, vals2])
        y_max_local = local_vals.max()
    
        # GLOBAL for scaling only
        h = yrange_global * 0.01
    
        if region == "mainland":
            # keep high (global)
            y = y_max_global + yrange_global * 0.08
        else:
            # 🔥 anchor to island data → MUCH lower
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
    
        # 🔥 KEY CHANGE: region-specific max
        local_vals = pd.concat([vals1, vals2])
        y_max_local = local_vals.max()
    
        # GLOBAL for scaling only
        h = yrange_global * 0.01
    
        if region == "mainland":
            # keep high (global)
            y = y_max_global + yrange_global * 0.08
        else:
            # 🔥 anchor to island data → MUCH lower
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
    figsize=(9.53 * cm, 7.51 * cm),
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
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
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
    # GLOBAL scaling
    # -------------------------
    all_vals = sub["ratio"]
    y_max_global = all_vals.max()
    y_min_global = all_vals.min()
    yrange_global = y_max_global - y_min_global

    # -------------------------
    # Statistical brackets
    # -------------------------
    for j, region in enumerate(order):

        x1 = j - 0.2
        x2 = j + 0.2

        vals1 = sub[(sub["region"] == region) & (sub["location"] == "in_roh")]["ratio"]
        vals2 = sub[(sub["region"] == region) & (sub["location"] == "out_roh")]["ratio"]

        if len(vals1) == 0 or len(vals2) == 0:
            continue

        y = y_max_global + yrange_global * 0.08
        h = yrange_global * 0.03

        ax.plot([x1, x1, x2, x2],
                [y, y + h, y + h, y],
                lw=0.6, c='black', zorder=3)

        # placeholder text (you can replace with *, **, etc.)
        ax.text(
            (x1 + x2) / 2,
            y + h + yrange_global * 0.02,
            "",
            ha='center',
            va='bottom',
            fontsize=6,
            fontproperties=arial_font
        )

    # -------------------------
    # Axis limits
    # -------------------------
    ax.set_ylim(
        y_min_global - yrange_global * 0.1,
        y_max_global + yrange_global * 0.3
    )

    # -------------------------
    # Labels (RESTORED)
    # -------------------------
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)

    if i == 0:
        ax.set_ylabel("Genetic load")
    else:
        ax.set_ylabel("")

    ax.set_xlabel("")
    ax.set_title(metric)

    # ticks styling
    ax.tick_params(axis='x', length=3, width=0.8)
    ax.tick_params(axis='y', length=3, width=0.8)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.18,
    right=0.95,
    top=0.90,
    bottom=0.12,
    wspace=0.12
)

# =========================
# Save
# =========================
plt.savefig("Figure_total_load_ROH_final.svg", format="svg", dpi=300)
plt.show()
```


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
variants = ["log_hom_del", "log_hom_lof"]   # <-- safer than .unique()

# =========================
# Loop panels
# =========================
for i, variant in enumerate(variants):

    ax = axes[i]

    sub = df_long[df_long["variant_type"] == variant]

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.grid(axis="y", visible=False)

    # Plot both regions
    for region in regions:
        d = sub[sub["Region"] == region]

        # Scatter
        ax.scatter(
            d["Froh_centered"],
            d["log_hom"],
            s=10,
            alpha=0.5,
            color=palette[region],
            edgecolor="none",
            zorder=1
        )

        # Regression
        if len(d) > 1:
            x = d["Froh_centered"].values
            y = d["log_hom"].values

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

    # Full box
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)

    # Panel title
    ax.set_title(variant, fontsize=7)

# =========================
# Axis labels
# =========================
for ax in axes:
    ax.set_xlabel(r"$F_{ROH}$ (centered)")

axes[0].set_ylabel("log(homozygotes)")

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
    ax.grid(axis="y", visible=False)

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

    # Full box
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)

    # Panel title
    ax.set_title(variant, fontsize=7)

# =========================
# Axis labels
# =========================
for ax in axes:
    ax.set_xlabel(r"$F_{ROH}$ (centered)")

axes[0].set_ylabel("homozygous derived")

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
variants = ["log_hom_del", "log_hom_lof"]   # <-- safer than .unique()

# =========================
# Loop panels
# =========================
for i, variant in enumerate(variants):

    ax = axes[i]

    sub = df_long[df_long["variant_type"] == variant]

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.grid(axis="y", visible=True)

    # Plot both regions
    for region in regions:
        d = sub[sub["Region"] == region]

        # Scatter
        ax.scatter(
            d["Froh_centered"],
            d["log_hom"],
            s=10,
            alpha=0.5,
            color=palette[region],
            edgecolor="none",
            zorder=1
        )

        # Regression
        if len(d) > 1:
            x = d["Froh_centered"].values
            y = d["log_hom"].values

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


```python

```


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
# Create a copy
df_freq = df_rohs_cor.copy()

# Loop over populations
for pop in sample_info["Population"].unique():
    inds = sample_info.loc[sample_info["Population"] == pop, "sample"]
    df_freq[pop] = df_rohs_cor[inds].mean(axis=1)

# Keep only relevant columns
pop_cols = sample_info["Population"].unique().tolist()
df_freq = df_freq[["WINDOW_ID", "chrom"] + pop_cols]

df_freq.head()
```


```python
corr_pop = df_freq[pop_cols].corr()
corr_pop
```


```python
# Map population → region
pop_to_region = dict(zip(sample_info["Population"], sample_info["Region"]))

# Compute region-level ROH frequency
for region in ["Mainland", "Island"]:
    pops = [p for p in pop_cols if pop_to_region[p] == region]
    df_freq[region] = df_freq[pops].mean(axis=1)

df_freq[["Mainland", "Island"]].head()
```


```python
corr_mainland_island = df_freq["Mainland"].corr(df_freq["Island"])
print("Mainland vs Island ROH correlation:", corr_mainland_island)
```


```python
import itertools

results = []

for pop1, pop2 in itertools.combinations(pop_cols, 2):
    
    r = df_freq[pop1].corr(df_freq[pop2])
    
    region1 = pop_to_region[pop1]
    region2 = pop_to_region[pop2]
    
    if region1 == region2:
        comparison = f"{region1}-{region1}"
    else:
        comparison = "Mainland-Island"
    
    results.append({
        "pop1": pop1,
        "pop2": pop2,
        "correlation": r,
        "comparison": comparison
    })

df_corr = pd.DataFrame(results)
df_corr.head()
```


```python
df_corr
```


```python
summary = df_corr.groupby("comparison")["correlation"].agg(
    mean="mean",
    sd="std",
    min="min",
    max="max",
    n="count"
)

summary
```


```python
import seaborn as sns

plt.figure(figsize=(8,6))
sns.boxplot(data=df_corr, x="comparison", y="correlation")
sns.stripplot(data=df_corr, x="comparison", y="correlation", color="black", alpha=0.4)

plt.title("ROH landscape similarity: within vs between regions")
plt.show()
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
def plot_roh_correct(df, value_col, title):

    chroms = sorted(df["chrom"].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    for i, chrom in enumerate(chroms):
        sub = df[df["chrom"] == chrom].sort_values("start")
        
        x = np.concatenate([sub["start"].values / 1e6,
                            [sub["end"].values[-1] / 1e6]])
        
        y = np.array([i - 0.4, i + 0.4])
        
        z = (sub[value_col].values * 100)[np.newaxis, :]

        ax.pcolormesh(
            x,
            y,
            z,
            cmap=cmap,
            norm=norm,
            shading='auto'
        )

    # axes
    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(range(1, len(chroms)+1))

    ax.set_xlabel("Position in Mb", fontsize=12)
    ax.set_ylabel("Chromosome", fontsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(False)

    # colorbar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label("% of individuals with ROH")

    # density inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="35%", height="30%", loc="upper right", borderpad=2)

    sns.kdeplot(df[value_col] * 100, fill=True, ax=axins, color="black")

    axins.set_xlabel("% with ROH", fontsize=8)
    axins.set_ylabel("Density", fontsize=8)
    axins.tick_params(axis='both', labelsize=8)

    for spine in ["top", "right"]:
        axins.spines[spine].set_visible(False)

    plt.title(title)
    plt.tight_layout()
    plt.show()
```


```python
plot_roh_correct(df_rohs_cor, "roh_all", "All individuals")

```


```python
plot_roh_correct(df_rohs_cor, "roh_mainland", "Mainland only")
```


```python
plot_roh_correct(df_rohs_cor, "roh_island", "Island only")
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
import numpy as np
import seaborn as sns

def plot_roh_correct(df, value_col, title):

    # ---- Global style ----
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    chroms = sorted(df["chrom"].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    # ---- ROH landscape ----
    for i, chrom in enumerate(chroms):
        sub = df[df["chrom"] == chrom].sort_values("start")
        
        x = np.concatenate([
            sub["start"].values / 1e6,
            [sub["end"].values[-1] / 1e6]
        ])
        
        y = np.array([i - 0.4, i + 0.4])
        
        z = (sub[value_col].values * 100)[np.newaxis, :]

        ax.pcolormesh(
            x,
            y,
            z,
            cmap=cmap,
            norm=norm,
            shading='auto'
        )

    # ---- Axes ----
    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(range(1, len(chroms)+1))

    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Chromosome")

    # clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(direction="out", length=4, width=1)

    # ---- Colorbar ----
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
        fraction=0.04
    )
    cbar.set_label("% individuals in ROH", rotation=90)
    cbar.ax.tick_params(labelsize=10)

    # ---- Density inset ----
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="32%", height="28%", loc="upper right", borderpad=2)

    sns.kdeplot(
        df[value_col] * 100,
        fill=True,
        ax=axins,
        color="black",
        linewidth=1
    )

    axins.set_xlabel("% ROH", fontsize=9)
    axins.set_ylabel("Density", fontsize=9)
    axins.tick_params(axis='both', labelsize=8)

    for spine in ["top", "right"]:
        axins.spines[spine].set_visible(False)

    # ---- Title ----
    ax.set_title(title, pad=10)

    plt.tight_layout()
    plt.show()
```


```python
plot_roh_correct(df_rohs_cor, "roh_mainland", "Mainland only")
```


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

# =========================
# USER-DEFINED FIGURE SIZE
# =========================
width_cm = 9.53
height_cm = 4.5

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
# Figure layout (2 panels)
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

        ax.pcolormesh(
            x,
            y,
            z,
            cmap=cmap,
            norm=norm,
            shading='auto'
        )

    # =========================
    # Axes styling (match your style)
    # =========================
    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(range(1, len(chroms)+1))

    ax.tick_params(axis='x', bottom=True, length=3, width=0.8, direction='out', pad=2)
    ax.tick_params(axis='y', left=True, length=3, width=0.8, direction='out', pad=2)

    # Full box
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)

    # Titles
    ax.set_title(region, fontsize=7)

# =========================
# Labels
# =========================
for ax in axes:
    ax.set_xlabel("Position (Mb)")

axes[0].set_ylabel("Chromosome")

# =========================
# Colorbar (shared)
# =========================
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    orientation='vertical',
    fraction=0.05,
    pad=0.02
)

cbar.set_label("% individuals in ROH", fontsize=7)
cbar.ax.tick_params(labelsize=6)

# =========================
# Layout tuning
# =========================
plt.subplots_adjust(
    left=0.12,
    right=0.88,
    top=0.88,
    bottom=0.20,
    wspace=0.15
)

# =========================
# Save
# =========================
plt.savefig("Figure_ROH_landscape_2panel.svg", format="svg", dpi=300)

plt.show()
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
def get_islands_deserts(df, col, pct=0.005):
    k = int(len(df) * pct)
    
    sorted_df = df.sort_values(col)
    
    islands = sorted_df.tail(k)
    
    # deserts = lowest non-zero
    nonzero = sorted_df[sorted_df[col] > 0]
    deserts = nonzero.head(k)
    
    return islands, deserts

islands_main, deserts_main = get_islands_deserts(df_rohs_cor, "roh_mainland")
islands_island, deserts_island = get_islands_deserts(df_rohs_cor, "roh_island")
```


```python
main_ids = set(islands_main["WINDOW_ID"])
island_ids = set(islands_island["WINDOW_ID"])

print("Shared islands:", len(main_ids & island_ids))
print("Island-specific islands:", len(island_ids - main_ids))
print("Mainland-specific islands:", len(main_ids - island_ids))

main_desert_ids = set(deserts_main["WINDOW_ID"])
island_desert_ids = set(deserts_island["WINDOW_ID"])

print("Shared deserts:", len(main_desert_ids & island_desert_ids))
print("Island-specific islands:", len(island_desert_ids - main_desert_ids))
print("Mainland-specific islands:", len(main_desert_ids - island_desert_ids))
```


```python
high_main = df_rohs_cor["roh_mainland"] > 0.8
high_island = df_rohs_cor["roh_island"] > 0.8

shared = sum(high_main & high_island)
main_only = sum(high_main & ~high_island)
island_only = sum(high_island & ~high_main)

print("Shared high ROH:", shared)
print("Mainland only:", main_only)
print("Island only:", island_only)
```


```python
import numpy as np

corr = np.corrcoef(df_rohs_cor["roh_mainland"],
                   df_rohs_cor["roh_island"])[0,1]

print("Correlation:", corr)
```


```python
plt.hist(df_rohs_cor["roh_mainland"], bins=50, alpha=0.5, label="Mainland")
plt.hist(df_rohs_cor["roh_island"], bins=50, alpha=0.5, label="Island")

plt.xlabel("ROH frequency")
plt.ylabel("Number of windows")
plt.legend()

plt.show()
```


```python

```


```python
def get_population(sample):
    pop = ''.join(re.findall(r'[A-Z]+', sample))
    pop = re.sub(r'^24', '', pop)
    if pop in ["CV", "FV"]:
        pop = "RG"
    return pop

sample_info = pd.DataFrame({"sample": sample_cols})
sample_info["Population"] = sample_info["sample"].apply(get_population)

sample_info.head()
```


```python
pop_groups = sample_info.groupby("Population")["sample"].apply(list)

pop_groups
```


```python
df_rohs_cor[sample_cols] = df_rohs_cor[sample_cols].apply(pd.to_numeric, errors='coerce')
for pop, samples in pop_groups.items():
    df_rohs_cor[f"roh_{pop}"] = (df_rohs_cor[samples] > 0).mean(axis=1)

# check
df_rohs_cor.filter(like="roh_").head()
```


```python
coords = df_rohs_cor["WINDOW_ID"].str.extract(r":(\d+)-(\d+)")

df_rohs_cor["start"] = coords[0].astype(int)
df_rohs_cor["end"] = coords[1].astype(int)
df_rohs_cor["mid"] = (df_rohs_cor["start"] + df_rohs_cor["end"]) / 2 / 1e6
```


```python
def plot_roh(df, value_col, title):

    chroms = sorted(df["chrom"].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    for i, chrom in enumerate(chroms):
        sub = df[df["chrom"] == chrom].sort_values("start")
        
        x = np.concatenate([sub["start"].values / 1e6,
                            [sub["end"].values[-1] / 1e6]])
        
        y = np.array([i - 0.4, i + 0.4])
        
        z = (sub[value_col].values * 100)[np.newaxis, :]

        ax.pcolormesh(
            x,
            y,
            z,
            cmap=cmap,
            norm=norm,
            shading='auto'
        )

    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(range(1, len(chroms)+1))

    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Chromosome")
    ax.set_title(title)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(False)

    # colorbar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label("% of individuals with ROH")

    # inset density (top right)
    axins = inset_axes(ax, width="30%", height="25%", loc="upper right", borderpad=1.5)
    sns.kdeplot(df[value_col] * 100, fill=True, ax=axins, color="black")

    axins.set_xlabel("% ROH", fontsize=8)
    axins.set_ylabel("Density", fontsize=8)
    axins.tick_params(labelsize=8)

    for spine in ["top", "right"]:
        axins.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()
```


```python
for pop in pop_groups.index:
    plot_roh(df_rohs_cor, f"roh_{pop}", f"Population: {pop}")
```


```python
def get_islands(df, col, pct=0.005):
    k = int(len(df) * pct)
    return df.sort_values(col).tail(k)
```


```python
set_SC = set(islands_per_pop["SC"]["WINDOW_ID"])
set_PK = set(islands_per_pop["PK"]["WINDOW_ID"])

print("Shared:", len(set_SC & set_PK))
print("SC only:", len(set_SC - set_PK))
print("PK only:", len(set_PK - set_SC))
```


```python
set_PM = set(islands_per_pop["PM"]["WINDOW_ID"])
set_PK = set(islands_per_pop["PK"]["WINDOW_ID"])

print("Shared:", len(set_PM & set_PK))
print("PM only:", len(set_PM - set_PK))
print("PK only:", len(set_PK - set_PM))
```


```python

```


```python
import seaborn as sns

def plot_individual_correlation(df, samples, title):
    
    mat = (df[samples] > 0).astype(int)
    
    corr = mat.corr()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="viridis")
    
    plt.title(title)
    plt.show()
```


```python
for pop, samples in pop_groups.items():
    plot_individual_correlation(df_rohs_cor, samples, f"{pop} correlation")
```


```python
from sklearn.metrics.pairwise import cosine_similarity

def plot_cosine(df, samples, title):
    
    mat = (df[samples] > 0).astype(int).T
    
    sim = cosine_similarity(mat)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(sim, cmap="viridis")
    
    plt.title(title + " (cosine)")
    plt.show()
```


```python
for pop, samples in pop_groups.items():
    plot_cosine(df_rohs_cor, samples, f"{pop} correlation")
```


```python
def shared_roh_fraction(df, samples):
    
    mat = (df[samples] > 0).astype(int)
    
    freq = mat.mean(axis=1)
    
    shared_all = (freq == 1).mean()
    shared_90 = (freq > 0.90).mean()
    shared_75 = (freq > 0.75).mean()
    shared_half = (freq > 0.5).mean()
    shared_25 = (freq > 0.25).mean()
    shared_none = (freq == 0).mean()

    
    return shared_all, shared_90, shared_75, shared_half, shared_25, shared_none
```


```python
for pop, samples in pop_groups.items():
    shared_all, shared_90, shared_75, shared_half, shared_25, shared_none = shared_roh_fraction(df_rohs_cor, samples)
    
    print(f"{pop}")
    print("  Shared by ALL:", shared_all)
    print("  Shared by >90%:", shared_90)
    print("  Shared by >75%:", shared_75)
    print("  Shared by >50%:", shared_half)
    print("  Shared by >25%:", shared_25)
    print("  Shared by NONE:", shared_none)
```


```python
import numpy as np

for pop1 in pop_groups.index:
    for pop2 in pop_groups.index:
        if pop1 >= pop2:
            continue
        
        r = np.corrcoef(
            df_rohs_cor[f"roh_{pop1}"],
            df_rohs_cor[f"roh_{pop2}"]
        )[0,1]
        
        
        print(f"{pop1} vs {pop2}: {r:.3f}")
```


```python

```


```python
for pop, samples in pop_groups.items():
    core = core_roh_regions(df_rohs_cor, samples)
    print(pop, len(core))
```


```python
df_rohs_cor[["roh_PK", "roh_PM", "roh_SC"]].head(20)
```


```python
df_rohs_cor[pop_groups["PK"]].head(20)
```


```python

```


```python

```
