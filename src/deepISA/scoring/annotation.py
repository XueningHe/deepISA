import pandas as pd
import os
import pyBigWig
import bioframe as bf
from loguru import logger

def subset_by_rna(df, expressed_tfs):
    """Mandatory hard filter for RNA evidence."""
    if df.empty:
        return df
    if expressed_tfs is None:
        return df
    # Handle dimeric names (e.g., GATA1::TAL1)
    prots = df["tf"].str.split("::", expand=True)
    p1 = prots[0].str.upper()
    p2 = prots.iloc[:, -1].fillna(prots[0]).str.upper()
    # Both parts of a dimer must be in the expressed list
    mask = p1.isin(expressed_tfs) & p2.isin(expressed_tfs)
    return df[mask].copy()



def check_remap(motif_df, remap_ref, region_tuple):
    """Checks if TFs in motif_df have overlapping ChIP peaks in the specific region."""
    if motif_df.empty:
        motif_df["remap_evidence"] = pd.Series(dtype=bool)
        return motif_df

    chrom, start, end = region_tuple
    # Filter ReMap peaks to only those within the current genomic window
    local_peaks = bf.select(remap_ref, (chrom, start, end))
    chip_tfs = set(local_peaks["TF"].unique())

    # Check dimer components against available ChIP TFs
    prots = motif_df["tf"].str.split("::", expand=True)
    p1 = prots[0].str.upper()
    p2 = prots.iloc[:, -1].fillna(prots[0]).str.upper()
    motif_df["remap_evidence"] = p1.isin(chip_tfs) & p2.isin(chip_tfs)
    return motif_df



class JasparAnnotator:
    def __init__(self, jaspar_path, expressed_tfs=None, score_thresh=500, remap_path=None):
        self.jaspar = pyBigWig.open(jaspar_path)
        self.score_thresh = score_thresh
        if expressed_tfs is not None:
            self.expressed_tfs = set(str(tf).upper() for tf in expressed_tfs)
        else:
            self.expressed_tfs = None
        # Load ReMap if provided
        self.remap_ref = None
        if remap_path:
            df = pd.read_csv(remap_path, sep='\t', header=None, usecols=[0, 1, 2, 3],
                             names=['chrom', 'start', 'end', 'detail'])
            # Robust extraction: split by colon or underscore
            df['TF'] = df['detail'].str.split('[:_]', expand=True)[0].str.upper()
            self.remap_ref = df[['chrom', 'start', 'end', 'TF']]

    def _get_motifs_in_region(self, region_tuple):
        """Fetches and parses motifs, ensuring they are strictly within bounds."""
        chrom, start, end = region_tuple
        try:
            entries = self.jaspar.entries(chrom, start, end)
        except:
            return pd.DataFrame()
        
        if not entries:
            return pd.DataFrame()

        df = pd.DataFrame(entries, columns=['start', 'end', 'details'])
        # Requirement: Motif must lie COMPLETELY within the given region
        df = df[(df['start'] >= start) & (df['end'] <= end)].copy()
        if df.empty:
            return df

        # Robust Parsing
        split_cols = df['details'].str.split('\t', expand=True)
        df['tf'] = split_cols[3].str.upper()
        df['score'] = pd.to_numeric(split_cols[1], errors='coerce').fillna(0).astype(int)
        df['strand'] = split_cols[2]
        df['chrom'] = chrom
        df['region'] = f"{chrom}:{start}-{end}"
        # Filtering
        df = df[df['score'] >= self.score_thresh]
        df = subset_by_rna(df, self.expressed_tfs)
        return df.drop_duplicates().reset_index(drop=True)

    def annotate(self, regions, outpath):
        """Streams motifs to disk."""
        if os.path.exists(outpath):
            os.remove(outpath)
        for i, (_, row) in enumerate(regions.iterrows()):
            if i % 1000 == 0:
                logger.info(f"Processing region {i}...")
            reg_tuple = (row['chrom'], row['start'], row['end'])
            df = self._get_motifs_in_region(reg_tuple)
            if df.empty:
                continue
            # Optional ReMap column
            if self.remap_ref is not None:
                df = check_remap(df, self.remap_ref, reg_tuple)
            # Final Column Management: chrom, start, end must be first
            cols = ['chrom', 'start', 'end', 'tf', 'score', 'strand', 'region']
            if "remap_evidence" in df.columns:
                cols.append("remap_evidence")
            df = df[cols]
            header = not os.path.exists(outpath)
            df.to_csv(outpath, index=False, mode='a', header=header)




def map_motifs(regions_df, 
               jaspar_path, 
               outpath, 
               expressed_tfs=None, 
               remap_path=None, 
               score_thresh=500):
    """High-level API for motif mapping."""
    logger.info("Starting motif mapping.")
    annotator = JasparAnnotator(
        jaspar_path=jaspar_path,
        expressed_tfs=expressed_tfs,
        score_thresh=score_thresh,
        remap_path=remap_path
    )
    
    annotator.annotate(regions_df, outpath)
    logger.info(f"Complete. Output saved to: {outpath}")
    return outpath