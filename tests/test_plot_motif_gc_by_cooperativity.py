import pytest
import pandas as pd
from unittest.mock import patch

# Mocking the data resource helper and the plotting function
@patch('deepISA.plotting.tf.get_data_resource')
@patch('deepISA.utils.plot_violin_with_statistics')
def test_plot_motif_gc_by_cooperativity_logic(mock_plot, mock_get_resource, tmp_path):
    """
    Tests the logic of calculating GC content from PFMs and merging with cooperativity.
    """
    # 1. Create a dummy JASPAR PFM file
    # TF1: High GC (All C and G)
    # TF2: Low GC (All A and T)
    jaspar_content = (
        ">MA0000.1\tTF_HIGH_GC\n"
        "A  [ 0 0 ]\n"
        "C  [ 10 10 ]\n"
        "G  [ 10 10 ]\n"
        "T  [ 0 0 ]\n"
        ">MA0001.1\tTF_LOW_GC\n"
        "A  [ 10 10 ]\n"
        "C  [ 0 0 ]\n"
        "G  [ 0 0 ]\n"
        "T  [ 10 10 ]\n"
    )
    jaspar_file = tmp_path / "dummy_jaspar.txt"
    jaspar_file.write_text(jaspar_content)
    
    # Configure the mock to return our local dummy file
    mock_get_resource.return_value = str(jaspar_file)

    # 2. Setup the cooperativity profile (df_tf)
    # Note: TF names are converted to upper() in the function logic
    df_tf = pd.DataFrame({
        'tf': ['TF_HIGH_GC', 'TF_LOW_GC'],
        'cooperativity': ['Independent', 'Synergistic']
    })
    
    # 3. Import and run the function
    from deepISA.plotting.tf import plot_motif_gc_by_cooperativity
    
    outpath = str(tmp_path / "test_plot.pdf")
    plot_motif_gc_by_cooperativity(df_tf, outpath=outpath)

    # 4. Verifications
    assert mock_plot.called
    args, _ = mock_plot.call_args
    passed_df = args[1]
    
    # Check that GC column exists
    assert 'GC' in passed_df.columns
    
    # Verify GC calculations
    # TF_HIGH_GC should be 1.0 (20 G/C counts / 20 total counts)
    # TF_LOW_GC should be 0.0 (0 G/C counts / 20 total counts)
    high_gc_val = passed_df.loc[passed_df['tf'] == 'TF_HIGH_GC', 'GC'].values[0]
    low_gc_val = passed_df.loc[passed_df['tf'] == 'TF_LOW_GC', 'GC'].values[0]
    
    assert high_gc_val == pytest.approx(1.0)
    assert low_gc_val == pytest.approx(0.0)
    
    # Verify the category ordering and type
    assert passed_df['cooperativity'].dtype.name == 'category'
    expected_order = ["Independent", "Redundant", "Intermediate", "Synergistic"]
    assert list(passed_df['cooperativity'].cat.categories) == expected_order
    
    
    