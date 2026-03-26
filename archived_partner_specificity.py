import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['pdf.fonttype']=42


cell_line="hepg2"

df=pd.read_csv(f"/isdata/alab/people/pcr980/DeepCompare/Pd7_TF_cooperativity/tf_pair_cooperativity_index_{cell_line}.csv")
df=df[df["c_sum"]>1].reset_index(drop=True)

# get tf type
tfs_codependent=pd.read_csv(f"/isdata/alab/people/pcr980/DeepCompare/Pd7_TF_cooperativity/tfs_codependent_{cell_line}.txt",header=None).iloc[:,0].tolist()
tfs_redundant=pd.read_csv(f"/isdata/alab/people/pcr980/DeepCompare/Pd7_TF_cooperativity/tfs_redundant_{cell_line}.txt",header=None).iloc[:,0].tolist()


df["protein2_cooperativity"]="other"
df.loc[df["protein2"].isin(tfs_codependent),"protein2_cooperativity"]="codependent"
df.loc[df["protein2"].isin(tfs_redundant),"protein2_cooperativity"]="redundant"


df_codependent=df[df["protein2_cooperativity"]=="codependent"].reset_index(drop=True)
df_codependent=df_codependent[["protein1","protein2","c_codependency"]]
df_codependent.rename(columns={"c_codependency":"cooperativity_index"},inplace=True)

df_redundant=df[df["protein2_cooperativity"]=="redundant"].reset_index(drop=True)
df_redundant=df_redundant[["protein1","protein2","c_redundancy"]]
df_redundant.rename(columns={"c_redundancy":"cooperativity_index"},inplace=True)
df_redundant["cooperativity_index"]=df_redundant["cooperativity_index"].abs()



def calc_top_5_percentage(df,tf):
    df_sub=df[df["protein2"]==tf].reset_index(drop=True)
    df_sub=df_sub[["protein2","cooperativity_index"]]
    if df_sub.shape[0]<10:
        return None
    df_topx_percent=df_sub.sort_values(by="cooperativity_index",ascending=False).head(df_sub.shape[0]//20)
    topx_sum=df_topx_percent["cooperativity_index"].sum()
    total_sum=df_sub["cooperativity_index"].sum()
    return topx_sum/total_sum



top_5_percentage_codependent=[]

for protein in df_codependent["protein2"].unique():
    top_5_percentage_codependent.append(calc_top_5_percentage(df_codependent,protein))

top_5_percentage_redundant=[]

for protein in df_redundant["protein2"].unique():
    top_5_percentage_redundant.append(calc_top_5_percentage(df_redundant,protein))


df_res=pd.DataFrame({"top_5_percentage":top_5_percentage_codependent+top_5_percentage_redundant,
                     "type":["codependent"]*len(top_5_percentage_codependent)+["redundant"]*len(top_5_percentage_redundant)})
# plot the results
plt.figure(figsize=(10,5))
sns.histplot(data=df_res,x="top_5_percentage",hue="type",kde=True,common_norm=False)
plt.savefig(f"partner_specificity_{cell_line}.png")
plt.close()
