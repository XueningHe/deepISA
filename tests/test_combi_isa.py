import pytest
import pandas as pd
import numpy as np
import torch
import os
from deepISA.modeling.cnn import Conv
from deepISA.scoring.combi_isa import (
    run_combi_isa, 
    calc_cooperativity_stats, 
    assign_cooperativity, 
    _filter_valid_interaction
)
from deepISA.utils import find_available_gpu

# ------------------------------------------------------------------
# 1. Test Data Generation (run_combi_isa)
# ------------------------------------------------------------------

def test_run_combi_isa_flow(tmp_path):
    """Verifies that the ISA generation writes raw values to CSV."""
    model_path = "data/mini_model.pt"
    out_path = tmp_path / "test_results.csv"
    
    model = Conv() 
    device = find_available_gpu()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    motif_locs = pd.DataFrame({
        "region": ["chr1:100-700", "chr1:100-700"],
        "tf": ["GATA1", "TAL1"],
        "start": [400, 500],
        "end": [415, 515]
    })
    mock_fasta = {"chr1": "A" * 2000}

    run_combi_isa(
        model=model,
        fasta=mock_fasta,
        motif_locs=motif_locs,
        device=device,
        outpath=str(out_path),
        track_idx=0
    )

    assert os.path.exists(out_path)
    res_df = pd.read_csv(out_path)
    # Check for the raw interaction column
    assert "interaction_t0" in res_df.columns
    assert not res_df["interaction_t0"].isna().all()

# ------------------------------------------------------------------
# 2. Test Statistical Logic (calc_cooperativity_stats)
# ------------------------------------------------------------------

@pytest.fixture
def mock_interaction_data():
    """Creates a mock dataset with Synergy, Redundancy, and Noise distributions."""
    np.random.seed(42)
    
    # Create a background noise (Null) distribution: centered at 0
    null_dist = np.random.normal(0, 0.01, 100) 
    
    # Synergistic Pair: Shifted positive
    syn_dist = np.random.normal(0.5, 0.05, 30)
    # Redundant Pair: Shifted negative
    red_dist = np.random.normal(-0.5, 0.05, 30)
    # Independent Pair: Overlaps with noise
    ind_dist = np.random.normal(0, 0.01, 30)
    # Intermediate Pair: Centered at 0 but high variance (fat tail)
    int_dist = np.random.normal(0, 0.2, 30)

    def create_rows(tf1, tf2, dist_vals, distance=50):
        return pd.DataFrame({
            "tf1": [tf1] * len(dist_vals),
            "tf2": [tf2] * len(dist_vals),
            "distance": [distance] * len(dist_vals),
            "interaction_t0": dist_vals,
            "isa1_t0": [1.0] * len(dist_vals), # Pass _filter_valid_interaction
            "isa2_t0": [1.0] * len(dist_vals),
            "isa_both_t0": [1.5] * len(dist_vals),
            "pair": [f"{tf1}|{tf2}"] * len(dist_vals)
        })

    # Null rows (Distance 150)
    null_rows = create_rows("N", "N", null_dist, distance=150)
    # Active rows (Distance 50)
    syn_rows = create_rows("TF_S", "P1", syn_dist)
    red_rows = create_rows("TF_R", "P2", red_dist)
    ind_rows = create_rows("TF_I", "P3", ind_dist)
    int_rows = create_rows("TF_M", "P4", int_dist)

    return pd.concat([null_rows, syn_rows, red_rows, ind_rows, int_rows], ignore_index=True)




def test_tf_pair_level_stats(mock_interaction_data, tmp_path):
    """Tests the tf_pair aggregation logic by verifying the output file."""
    out_path = tmp_path / "stats_out.csv"
    # We ignore the return value (results will be None)
    calc_cooperativity_stats(
        mock_interaction_data,
        track_idx=0,
        outpath=out_path, 
        level='tf_pair'
    )
    # 1. Confirm the file exists
    assert os.path.exists(out_path)
    # 2. Load the file to verify the contents
    results = pd.read_csv(out_path)
    # 3. Verify Synergy in the saved file
    # Note: 'TF_S|P1' comes from the mock data fixture
    syn_row = results[results['tf_pair'] == "TF_S|P1"].iloc[0]
    assert syn_row['cooperativity'] == "Synergistic"
    assert syn_row['ks_q'] < 0.1

def test_tf_level_stats(mock_interaction_data, tmp_path):
    """Tests that TFs are evaluated based on their pooled interactions in the saved file."""
    out_path = tmp_path / "tf_stats_out.csv"
    
    calc_cooperativity_stats(
        mock_interaction_data,
        track_idx=0,
        outpath=out_path,
        level='tf'
    )
    
    # 1. Confirm file exists
    assert os.path.exists(out_path)
    
    # 2. Load and verify
    results = pd.read_csv(out_path)
    tf_s_row = results[results['tf'] == "TF_S"].iloc[0]
    assert tf_s_row['cooperativity'] == "Synergistic"
    assert tf_s_row['count'] == 30

# ------------------------------------------------------------------
# 3. Test Assignment Logic (assign_cooperativity)
# ------------------------------------------------------------------

def test_assign_cooperativity_gates():
    """Tests the logic gates for each category using manually defined Q-values."""
    data = {
        "name": ["S", "R", "M", "I"],
        "ks_q": [0.001, 0.001, 0.001, 0.5], # Significant, Significant, Significant, Noise
        "coop_score": [0.9, -0.9, 0.0, 0.9], # Pos, Neg, Mid, (Ignored)
    }
    df = pd.DataFrame(data)
    
    result = assign_cooperativity(df, q_val_thresh=0.1, synergy_thresh=0.3, redun_thresh=-0.3)
    
    assert result.loc[result["name"] == "S", "cooperativity"].iloc[0] == "Synergistic"
    assert result.loc[result["name"] == "R", "cooperativity"].iloc[0] == "Redundant"
    assert result.loc[result["name"] == "M", "cooperativity"].iloc[0] == "Intermediate"
    assert result.loc[result["name"] == "I", "cooperativity"].iloc[0] == "Independent"

# ------------------------------------------------------------------
# 4. Test Filtering logic
# ------------------------------------------------------------------

def test_filter_valid_interaction():
    """Ensures non-activators or invalid KOs are dropped."""
    data = {
        "isa1_t0": [1.0, -0.5, 1.0], # Valid, Negative (Repressor), Valid
        "isa2_t0": [1.0, 1.0, 1.0],  # Valid, Valid, Valid
        "isa_both_t0": [1.5, 1.0, 0.8], # Valid (1.5>1.0), Valid, Invalid (0.8 < 1.0)
        "interaction_t0": [0.5, 0.5, 0.5]
    }
    df = pd.DataFrame(data)
    
    filtered = _filter_valid_interaction(df, track_idx=0)
    
    # Only the first row should survive
    assert len(filtered) == 1
    assert filtered["isa1_t0"].iloc[0] == 1.0