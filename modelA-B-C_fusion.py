## Steps
# * 1-Import and combine datasets from mallet (scoresA), setsdg (scoresB), sdgmapper (scoresC)
# * 2-compute scores, normalize, compute cognitive diversity
# * 3-compute breaks on the cognitive diversity series for decision rule
# * 4-determine the final classification using score or rank combination decision rule


# ### Preamble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt  # to identify breaks in the series
import random

pd.set_option("display.float_format", lambda x: "%.6f" % x)
# set paths
in_path = "/Users/mlafleur/Projects/SDGfusion/input/"  # where the output of each classifier is placed
out_path = "/Users/mlafleur/Projects/SDGfusion/output/"  # Results are here

# ### Loading the dataset

# Load and prepare data from Model A, Model B, Model C

##MODEL A (SDGClassy)
# load the classification scores for model A

dfa = pd.read_csv(
    in_path + "output_a.csv",
    sep="\t",
    header=None,
    skiprows=[0],
    float_precision="round_trip",
    names=[
        "doc",
        "topic0",
        "topic1",
        "topic2",
        "topic3",
        "topic4",
        "topic5",
        "topic6",
        "topic7",
        "topic8",
        "topic9",
        "topic10",
        "topic11",
        "topic12",
        "topic13",
        "topic14",
        "topic15",
        "topic16",
        "topic17",
    ],
)
dfa = dfa.astype(object)
# removing unneccesary strings in the filename, which will be used to match later
dfa["doc"] = dfa["doc"].str.replace(
    "file:/Users/mlafleur/Projects/setsdg/input_cln/", ""
)  # need a more robust search
dfa["doc"] = dfa["doc"].str.replace(".txt", "")
dfa = dfa[dfa.doc != ".DS_Store"]

# mapping to sdgs (scoresA only)
map = pd.read_csv(in_path + "topic-sdg_mapping.csv", sep=",")  # load the map
dfa = dfa.rename(columns=map.set_index("topic")["sdg"].to_dict())
dfa.drop("delete", axis=1, inplace=True)  # one of the topics is non-sdg and is deleted

# re-weighting scores to add to 1 since we dropped an observation
dfa = dfa.set_index("doc")
res = dfa.div(dfa.sum(axis=1), axis=0)
res = res.reset_index()

# from wide to long
df_long_a = pd.melt(res, id_vars=["doc"], var_name="sdg", value_name="score_a")
df_long_a["score_a"] = pd.to_numeric(df_long_a["score_a"])
df_long_a["sdg"] = pd.to_numeric(df_long_a["sdg"])
df_long_a = df_long_a.sort_values(["doc", "sdg"])


##MODEL B (Linked SDG)
# load the classification scores for model B
dfb = pd.read_csv(
    in_path + "output_b.csv",
    sep="\t",
    header=None,
    float_precision="round_trip",
    names=[
        "doc",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
    ],
)

# removing unneccesary strings in the filename, which will be used to match later
dfb["doc"] = dfb["doc"].str.replace(".txt.pdf", "")
dfb = dfb[dfb.doc != ".DS_Store"]

# from wide to long
df_long_b = pd.melt(dfb, id_vars=["doc"], var_name="sdg", value_name="score_b")
df_long_b["sdg"] = pd.to_numeric(df_long_b["sdg"])
df_long_b = df_long_b.sort_values(["doc", "sdg"])


##MODEL C (sdgmapper)

dfc = pd.read_csv(
    in_path + "output_c.csv",
    header=None,
    skiprows=[0],
    float_precision="round_trip",
    names=[
        "doc",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
    ],
)

# removing unneccesary strings in the filename, which will be used to match later
dfc["doc"] = dfc["doc"].str.replace(".txt", "")
dfc["doc"] = dfc["doc"].str.replace("_0", "")
dfc = dfc[dfc.doc != ".DS_Store"]

# from wide to long
df_long_c = pd.melt(dfc, id_vars=["doc"], var_name="sdg", value_name="score_c")
df_long_c["sdg"] = pd.to_numeric(df_long_c["sdg"])
df_long_c["score_c"] = df_long_c["score_c"] / 100
df_long_c = df_long_c.sort_values(["doc", "sdg"])


##MERGE THE FILES
df_long = df_long_a.merge(df_long_b, on=["doc", "sdg"], how="outer")
df_long = df_long.merge(df_long_c, on=["doc", "sdg"], how="outer")
# df_long.query("score_a != score_a" or "score_b != score_b") #see if any scores are missing for a file
df_long = df_long.dropna(
    how="any", axis=0
)  # if files don't match, delete the observation

##COMPUTE weighted scores using max-min;  formula: y = (x – min) / (Max – min)
# Define the columns for which you want to compute normalized scores
columns_to_normalize = ["score_a", "score_b", "score_c"]

# Loop through each column and compute the normalized scores
for col in columns_to_normalize:
    grouper = df_long.groupby("doc")[col]
    maxes = grouper.transform("max")
    mins = grouper.transform("min")
    df_long[col + "_norm"] = (df_long[col] - mins) / (maxes - mins)


##COMPUTE RANK AND SCORE COMBINATION
# Score combination and SC rank
df_long["sc"] = (
    df_long["score_a"] + df_long["score_b"] + df_long["score_c"]
) / 3  # score combination
df_long["sc_rank"] = (
    df_long.groupby("doc")["sc"].rank(ascending=False).astype(int)
)  # rank of scores: higher is better

# Rank combination and RC rank
df_long["rank_a"] = (
    df_long.groupby("doc")["score_a"].rank(ascending=False).astype(int)
)  # rankA
df_long["rank_b"] = (
    df_long.groupby("doc")["score_b"].rank(ascending=False).astype(int)
)  # rankB
df_long["rank_c"] = (
    df_long.groupby("doc")["score_c"].rank(ascending=False).astype(int)
)  # rankC
df_long["rc"] = (
    df_long["rank_a"] + df_long["rank_b"] + df_long["rank_c"]
) / 3  # rank combination
df_long["rc_rank"] = (
    df_long.groupby("doc")["rc"].rank(ascending=True).astype(int)
)  # rank of rank combination: lower is better

# Generate a random factor
df_long["fuzz"] = [random.randint(-1000, 1000) for k in df_long.index]

# Alternative rank combination to avoid ties: sum of squares with preference to model A and B
df_long["rc2"] = (
    df_long["rank_a"] ** (2 + (df_long["fuzz"] / 1000000000))
    + df_long["rank_b"] ** (2 + (df_long["fuzz"] / 1000000000000))
    + df_long["rank_c"] ** 2
)
df_long["rc_rank2"] = (
    df_long.groupby("doc")["rc2"].rank(ascending=True).astype(int)
)  # alternative rank of rank combination: lower is better


# Save the file
df_long.to_csv(out_path + "scores_long.csv", index=False, header=True)


## Rank score function - using normalized scores
# get data
df_long = pd.read_csv(
    out_path + "scores_long.csv", sep=",", float_precision="round_trip"
)

# Need to re-order score_a_norm, score_b_norm, score_c_norm individually to build RSC (see Frank's method)
# For each, we copy the data, reorder, and then join them together
rsc_a = df_long.copy()
drop_a = [
    "rc",
    "sc",
    "score_a",
    "score_b",
    "score_c",
    "score_b_norm",
    "score_c_norm",
    "sc_rank",
    "rank_a",
    "rank_b",
    "rank_c",
    "rc_rank",
]
rsc_a = rsc_a.drop(columns=drop_a)
rsc_a = (
    rsc_a.sort_values(["doc", "score_a_norm"], ascending=False).groupby("doc").head(17)
)  # head(17) returns first 17 obs of each group
rsc_a = rsc_a.reset_index(drop=True)
rsc_a["index1"] = rsc_a.index

rsc_b = df_long.copy()
drop_b = [
    "rc",
    "sc",
    "score_a",
    "score_b",
    "score_c",
    "score_a_norm",
    "score_c_norm",
    "sc_rank",
    "rank_a",
    "rank_b",
    "rank_c",
    "rc_rank",
]
rsc_b = rsc_b.drop(columns=drop_b)
rsc_b = (
    rsc_b.sort_values(["doc", "score_b_norm"], ascending=False).groupby("doc").head(17)
)
rsc_b = rsc_b.reset_index(drop=True)
rsc_b["index1"] = rsc_b.index

rsc_c = df_long.copy()
drop_c = [
    "rc",
    "sc",
    "score_a",
    "score_b",
    "score_c",
    "score_a_norm",
    "score_b_norm",
    "sc_rank",
    "rank_a",
    "rank_b",
    "rank_c",
    "rc_rank",
]
rsc_c = rsc_c.drop(columns=drop_c)
rsc_c = (
    rsc_c.sort_values(["doc", "score_c_norm"], ascending=False).groupby("doc").head(17)
)
rsc_c = rsc_c.reset_index(drop=True)
rsc_c["index1"] = rsc_c.index

# combine
rsc = pd.merge(rsc_a, rsc_b, on="index1")
rsc = pd.merge(rsc, rsc_c, on="index1")

##Compute measure of dispersion
# Use variance if there are more than 2 models
rsc["dispersion"] = rsc[["score_a_norm", "score_b_norm", "score_c_norm"]].var(
    axis=1
)  # variance

rsc["dispersion"] = rsc["dispersion"].replace(
    0, np.nan
)  # we will not use the zeros in the averages and medians

# Save
rsc.to_csv(out_path + "rsc_scores.csv", index=False, header=True)

##Reduce data to summary based on means and medias, variance, etc
# This will be used later to categorize each publication

# create means and medians of the squared errors for each publication
means = rsc.groupby(["doc_x"])["dispersion"].mean().reset_index()
means.rename(columns={"dispersion": "mean"}, inplace=True)

medians = rsc.groupby(["doc_x"])["dispersion"].median().reset_index()
medians.rename(columns={"dispersion": "median"}, inplace=True)

# create cognitive diversity: sqrto of the sum of squared differences
cog_div = rsc.groupby(["doc_x"])["dispersion"].sum().reset_index()
cog_div["cogdiv"] = np.sqrt(cog_div["dispersion"])
cog_div = cog_div.drop(columns="dispersion")

# combine
dispersion = pd.merge(means, cog_div, on="doc_x")
dispersion.rename(columns={"doc_x": "doc"}, inplace=True)

# Determine which quartile and label
dispersion["mean_quartile"] = pd.qcut(dispersion["mean"], labels=False, q=4)
dispersion["cogdiv_quartile"] = pd.qcut(dispersion["cogdiv"], labels=False, q=4)

# Save each publication's quartiles
dispersion.to_csv(out_path + "dispersion.csv", index=False, header=True)

##merge dispersion measure into rank and score data
final = pd.merge(
    df_long,
    dispersion[["doc", "cogdiv_quartile"]],  # using cogdiv_quartile here
    on="doc",
)

## Decision rule
# use dispersion to decide on using score or rank combination
# if quartile == 3, use rank combination to determine classification
# else, use score combination to determine classification
final.loc[final["cogdiv_quartile"] == 3, "final_rank"] = final["rc_rank"].astype(
    int
)  # use rank combination result if in top quartile
final.loc[final["cogdiv_quartile"] != 3, "final_rank"] = final["sc_rank"].astype(
    int
)  # use score combination result if not in the top quartile

drop_these = [
    "score_a",
    "score_b",
    "score_c",
    "score_a_norm",
    "score_b_norm",
    "score_c_norm",
    "rc",
    "sc",
    "rank_a",
    "rank_b",
    "rank_c",
    "rc",
    "fuzz",
]
final = final.drop(columns=drop_these)

# final.sort_values(by=['doc','final_rank'], inplace=True) #sort by final_rank

# Save scores
final.to_csv(out_path + "final.csv", index=False, header=True)

final.head(20)


# ------------------------
# ### Below this point, only for exploration and testing
# # Plot

"""
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import warnings
import altair as alt

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

##Make plot of the results from final.csv
# show top 3 or 5 SDGs? How to summarize the results?

dispersion = pd.read_csv(
    out_path + "dispersion.csv", sep=",", float_precision="round_trip"
)

dispersion.plot(kind="scatter", x="mean", y="cogdiv")

dispersion.boxplot(column=["mean", "cogdiv"])

dispersion.plot(kind="hist")
"""
