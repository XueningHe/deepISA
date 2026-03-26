import os
import pandas as pd
import pytest
from deepISA.validating.ppi import validate_ppi, validate_cofactor_recruitment, annotate_cofactor_recruitment

@pytest.fixture
def mock_results_df():
    """Creates a dummy cooperativity profile with enough rows to avoid pvalue warnings."""
    data = {
        "pair": [
            "GATA1|TAL1", "GATA1|TAL1", "GATA1|TAL1",  # Known PPI
            "FAKE1|FAKE2", "FAKE1|FAKE2", "FAKE1|FAKE2" # Unknown PPI
        ],
        "synergy_score": [0.8, 0.85, 0.78, 0.1, 0.12, 0.08],
        "independence_score": [0.2, 0.15, 0.22, 0.9, 0.88, 0.92]
    }
    df = pd.DataFrame(data)
    # The function expects 'reported_ppi' to be categorical after processing,
    # but initially, it just needs the 'pair' column.
    return df




def test_validate_ppi_new_signature(mock_results_df, tmp_path):
    """Test the refactored signature with custom outpath and labels."""
    plot_file = tmp_path / "custom_validation_plot.pdf"
    
    # Run validation
    result_df = validate_ppi(
        df=mock_results_df,
        score_col="synergy_score",
        outpath=str(plot_file),
        title="Custom Test Title",
        x_label="Experimental Evidence",
        y_label="Digital Synergy"
    )
    
    # 1. Check if file was created at the exact outpath
    assert plot_file.exists()
    
    # 2. Check if the column was added and is categorical
    assert "reported_ppi" in result_df.columns
    assert result_df["reported_ppi"].dtype == "category"
    assert set(result_df["reported_ppi"].cat.categories) == {"No", "Yes"}

def test_ppi_naming_normalization(tmp_path):
    """Verify that TF order (A|B vs B|A) doesn't break the PPI lookup."""
    # Create pairs that are swapped relative to standard alphabetizing
    df = pd.DataFrame({
        "pair": ["TAL1|GATA1", "TAL1|GATA1", "TAL1|GATA1"], 
        "synergy_score": [0.9, 0.88, 0.91]
    })
    
    plot_file = tmp_path / "swap_test.pdf"
    result = validate_ppi(
        df=df,
        score_col="synergy_score",
        outpath=str(plot_file)
    )
    
    # Because GATA1|TAL1 is in the reference, it should match even if input is TAL1|GATA1
    # Check the first row
    assert result.iloc[0]["reported_ppi"] == "Yes"

def test_invalid_score_col(mock_results_df, tmp_path):
    """Test behavior when a non-existent column is passed."""
    plot_file = tmp_path / "empty.pdf"
    
    # This should log a warning and return the df without crashing or plotting
    result = validate_ppi(
        df=mock_results_df,
        score_col="non_existent_column",
        outpath=str(plot_file)
    )
    
    assert not plot_file.exists()
    assert "reported_ppi" in result.columns

def test_ppi_plot_formatting(mock_results_df, tmp_path):
    """Ensure the function handles custom figure sizes and rotations."""
    plot_file = tmp_path / "formatted_plot.png"
    
    validate_ppi(
        df=mock_results_df,
        score_col="independence_score",
        outpath=str(plot_file),
        fig_size=(4, 4),
        rotation=45
    )
    
    assert plot_file.exists()
    
    
    
    
    
    
def test_dimer_exclusion(tmp_path):
    """Verify that any pair containing '::' (dimers) is removed from analysis."""
    # Create a mixture of valid pairs and dimers
    df = pd.DataFrame({
        "pair": [
            "GATA1|TAL1",        # Valid monomer|monomer
            "ARID3B|MAFG::NFE2L1" # Dimer (should be excluded)
        ],
        "synergy_score": [0.8, 1.2]
    })
    
    plot_file = tmp_path / "dimer_test.pdf"
    
    # Run the validation
    # We expect the dimer row to be dropped immediately
    result = validate_ppi(
        df=df,
        score_col="synergy_score",
        outpath=str(plot_file)
    )
    
    # 1. Check that the dimer row is gone
    assert len(result) == 1
    assert result.iloc[0]["pair"] == "GATA1|TAL1"
    assert not result["pair"].str.contains("::").any()
    
    # 2. Check that the valid row was correctly annotated
    # (Assuming GATA1|TAL1 is in your Human_TF_TF_I.txt)
    assert result.iloc[0]["reported_ppi"] == "Yes"
    
    
    
    
    
    
    
    
def test_annotate_cofactor_counting():
    """Verify 0, 1, 2 logic and dimer exclusion."""
    df = pd.DataFrame({
        "pair": ["GATA1|SOX2", "GATA1|FAKE", "FAKE1|FAKE2", "DIMER|A::B"],
        "synergy_score": [1.0, 0.5, 0.1, 0.0]
    })
    annotated = annotate_cofactor_recruitment(df, cofactors=["Mediator"])
    # Check dimer was removed
    assert len(annotated) == 3
    print(annotated)
    assert not annotated['pair'].str.contains("::").any()
    val_2 = annotated[annotated['pair'] == "GATA1|SOX2"]["count_Mediator"].iloc[0]
    val_0 = annotated[annotated['pair'] == "FAKE1|FAKE2"]["count_Mediator"].iloc[0]
    assert val_0 == 0
    assert val_2 == 2 # This is where your current failure is happening






def test_validate_cofactor_pipeline(tmp_path):
    """Test that the user can call one function to annotate and plot."""
    # AR and AKAP8 are 1 for POLII. GATA1 is 0.
    df = pd.DataFrame({
        "pair": ["AKAP8|AR", "AKAP8|GATA1", "GATA1|FAKE", "DIMER|A::B"],
        "synergy_score": [0.9, 0.7, 0.2, 0.5]
    })
    outpath = tmp_path / "result.pdf"
    # User only calls this
    result = validate_cofactor_recruitment(
        df=df,
        score_col="synergy_score",
        outpath=str(outpath),
        cofactor_name="POLII"
    )
    # 1. Verify Dimer Exclusion
    assert len(result) == 3
    # 2. Verify Annotation happened inside
    assert "count_POLII" in result.columns
    assert result[result["pair"] == "AKAP8|AR"]["count_POLII"].iloc[0] == 2
    # 3. Verify Plot created
    assert outpath.exists()





def test_validate_cofactor_plot_all_internal(tmp_path):
    """Test that passing None internally triggers batch plotting of all cofactors."""
    # Use TFs that have different cooperativity values in TF_Cof_I.txt
    # AR: Mediator=1, POLII=1
    # AKAP8: Mediator=0, POLII=1
    # GATA1: Mediator=0, POLII=0
    df = pd.DataFrame({
        "pair": [
            "AR|CDC5L",    # Mediator count 2, POLII count 2
            "AR|AKAP8",    # Mediator count 1, POLII count 2
            "AKAP8|GATA1", # Mediator count 0, POLII count 1
            "GATA1|FAKE"   # Mediator count 0, POLII count 0
        ],
        "synergy_score": [0.9, 0.8, 0.4, 0.2]
    })
    base_out = tmp_path / "batch.pdf"
    # This call now handles annotation AND plotting internally
    validate_cofactor_recruitment(
        df=df, 
        score_col="synergy_score", 
        outpath=str(base_out), 
        cofactor_name=None
    )
    # Now that we have variation (0, 1, and 2), these files must exist
    assert (tmp_path / "batch_POLII.pdf").exists()
    assert (tmp_path / "batch_Mediator.pdf").exists()
    assert (tmp_path / "batch_SWI_SNF.pdf").exists()