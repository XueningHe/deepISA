import pytest
import pandas as pd
import os
from deepISA.scoring.annotation import map_motifs

# Path to the resource prepared by the mock generator
MINI_JASPAR = "data/mini_jaspar.bb"

@pytest.fixture
def mock_evidence(tmp_path):
    """Creates mock TF list and ReMap file for testing."""
    # Mock Expressed TF List
    expressed_tfs = ["CTCF", "GATA1", "ETS1", "TAL1"]
    
    # Mock ReMap (ChIP): Only a peak for ETS1 on chr2
    remap_path = tmp_path / "remap_peaks.bed"
    # Format: chrom, start, end, detail (TF:info), x, strand, ts, te, score
    # Note: Using 'ETS1_HUMAN' to test the robust regex split logic
    remap_data = [
        ["chr2", 4000, 6000, "ETS1_HUMAN:mock_peak", ".", "+", 4000, 6000, 1000]
    ]
    pd.DataFrame(remap_data).to_csv(remap_path, sep='\t', index=False, header=False)
    return expressed_tfs, str(remap_path)

def test_jaspar_basic_annotation_and_order(tmp_path):
    """Test findings, region persistence, and column ordering (no ReMap)."""
    if not os.path.exists(MINI_JASPAR):
        pytest.skip("mini_jaspar.bb not found")
    
    regions = pd.DataFrame({'chrom': ['chr1'], 'start': [900], 'end': [1100]})
    expressed_tfs = ["CTCF"] 
    out_dir = tmp_path / "results"
    
    out_file = map_motifs(regions, MINI_JASPAR, expressed_tfs, str(out_dir), score_thresh=100)
    df = pd.read_csv(out_file)
    
    assert not df.empty
    assert df.iloc[0]['tf'] == "CTCF"
    assert df.iloc[0]['region'] == "chr1:900-1100"
    
    # Requirement: First 3 columns must be chrom, start, end
    assert list(df.columns[:3]) == ['chrom', 'start', 'end']
    
    # Verify Absence of remap_evidence and rna/is_expressed columns
    assert 'remap_evidence' not in df.columns
    assert 'rna_evidence' not in df.columns
    assert 'is_expressed' not in df.columns

def test_mandatory_rna_filtering(tmp_path):
    """Test that TFs NOT in the expressed_tfs list are strictly filtered out."""
    expressed_tfs = ["CTCF"] # ETS1 is NOT here
    # Region 2 contains ETS1 in the JASPAR file
    regions = pd.DataFrame({'chrom': ['chr2'], 'start': [4000], 'end': [6000]})
    
    out_dir = tmp_path / "rna_filter_test"
    out_file = map_motifs(regions, MINI_JASPAR, expressed_tfs, str(out_dir), score_thresh=100)
    
    # File should contain no rows because ETS1 was not in expressed_tfs
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        assert df.empty

def test_remap_dynamic_column_and_logic(tmp_path, mock_evidence):
    """Test that remap_evidence appears only when path is provided and labels correctly."""
    expressed_tfs, remap_p = mock_evidence
    
    # Region 1: chr1 (CTCF) - No ReMap peak in our mock
    # Region 2: chr2 (ETS1) - Has ReMap peak
    regions = pd.DataFrame({
        'chrom': ['chr1', 'chr2'],
        'start': [900, 4000],
        'end': [1100, 6000]
    })
    
    out_dir = tmp_path / "remap_test"
    result_path = map_motifs(
        regions, MINI_JASPAR, expressed_tfs, str(out_dir), 
        remap_path=remap_p, score_thresh=100
    )
    df = pd.read_csv(result_path)
    
    # Verify both survived RNA filtering (expressed_tfs includes both)
    assert set(df['tf']) == {"CTCF", "ETS1"}
    
    # Verify dynamic column name "remap_evidence"
    assert "remap_evidence" in df.columns
    
    # Verify flag values (ETS1 has evidence, CTCF does not)
    assert df[df['tf'] == "CTCF"]['remap_evidence'].iloc[0] == False
    assert df[df['tf'] == "ETS1"]['remap_evidence'].iloc[0] == True

def test_strict_coordinate_containment(tmp_path):
    """Test that motifs partially outside the region are excluded."""
    expressed_tfs = ["CTCF"]
    # If a CTCF motif exists at 950-1050, it should be FOUND in 900-1100
    # but NOT in 1000-1100 (because it starts before the region)
    regions = pd.DataFrame({'chrom': ['chr1'], 'start': [1000], 'end': [1100]})
    
    out_dir = tmp_path / "boundary_test"
    out_file = map_motifs(regions, MINI_JASPAR, expressed_tfs, str(out_dir), score_thresh=100)
    
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        # Check that any motifs found are strictly within [1000, 1100]
        for _, row in df.iterrows():
            assert row['start'] >= 1000
            assert row['end'] <= 1100

def test_dimeric_motif_logic(tmp_path):
    """Ensures dimeric TFs (GATA1::TAL1) require BOTH subunits in expressed_tfs."""
    # Case 1: TAL1 missing - Dimer should be filtered
    expressed_tfs = ["GATA1", "CTCF"] 
    region = pd.DataFrame({'chrom': ['chr1'], 'start': [1900], 'end': [2100]})
    
    out_file = map_motifs(region, MINI_JASPAR, expressed_tfs, str(tmp_path/"dimer_fail"), score_thresh=100)
    
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        # Monomer GATA1 survives, Dimer GATA1::TAL1 is filtered
        assert "GATA1" in df['tf'].values
        assert "GATA1::TAL1" not in df['tf'].values

    # Case 2: Both present - Dimer should survive
    expressed_tfs = ["GATA1", "TAL1", "CTCF"]
    out_file_2 = map_motifs(region, MINI_JASPAR, expressed_tfs, str(tmp_path/"dimer_pass"), score_thresh=100)
    df_2 = pd.read_csv(out_file_2)
    assert "GATA1::TAL1" in df_2['tf'].values