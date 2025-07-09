import os
import pandas as pd
from argo.vocab import FragmentVocabulary

def test_basic_average():
    """Test 1: Basic loading with average scoring"""
    print('='*60)
    print('Test 1: Basic loading with average scoring')
    print('='*60)
    
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    
    # Load first 500 lines for testing
    df_test = pd.read_csv(test_csv, nrows=500)
    print(f'Loaded {len(df_test)} molecules for testing')
    
    # Test average scoring
    vocab = FragmentVocabulary(df_test, min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    df = vocab.to_dataframe()
    
    print('Average scoring results:')
    print(f'Vocabulary size: {len(df)}')
    print('Top 5 fragments:')
    print(df.head())
    
    assert not df.empty, 'Vocabulary DataFrame should not be empty.'
    print('✓ Test 1 passed: Basic average scoring works')

def test_basic_enrichment():
    """Test 2: Basic loading with enrichment scoring"""
    print('\n' + '='*60)
    print('Test 2: Basic loading with enrichment scoring')
    print('='*60)
    
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    
    # Load first 500 lines for testing
    df_test = pd.read_csv(test_csv, nrows=500)
    print(f'Loaded {len(df_test)} molecules for testing')
    
    # Test enrichment scoring
    vocab = FragmentVocabulary(df_test, scoring_method='enrichment', top_percent=5.0, 
                             min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    df = vocab.to_dataframe()
    
    print('Enrichment scoring results:')
    print(f'Vocabulary size: {len(df)}')
    print('Top 5 fragments:')
    print(df.head())
    
    assert not df.empty, 'Vocabulary DataFrame should not be empty.'
    print('✓ Test 2 passed: Basic enrichment scoring works')

def test_incremental_enrichment():
    """Test 3: Incremental updates with enrichment scoring"""
    print('\n' + '='*60)
    print('Test 3: Incremental updates with enrichment scoring')
    print('='*60)
    
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    
    # Load all data to split
    all_data = pd.read_csv(test_csv)
    print(f'Total data available: {len(all_data)} molecules')
    
    # Split into parts
    df_part1 = all_data.iloc[:300].copy()  # First 300
    df_part2 = all_data.iloc[300:700].copy()  # Next 400
    df_part3 = all_data.iloc[700:1000].copy()  # Next 300
    
    print(f'Part 1: {len(df_part1)} molecules')
    print(f'Part 2: {len(df_part2)} molecules')
    print(f'Part 3: {len(df_part3)} molecules')
    
    # Step 1: Load first 300 lines with average scoring
    print('\n--- Step 1: Initial load with average scoring ---')
    vocab = FragmentVocabulary(df_part1, scoring_method='average', 
                             min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    df1 = vocab.to_dataframe()
    print(f'Initial vocabulary size: {len(df1)}')
    print('Top 5 fragments (average scoring):')
    print(df1.head())
    
    # Step 2: Add next 400 lines and switch to enrichment scoring
    print('\n--- Step 2: Add 400 more molecules and switch to enrichment ---')
    vocab.add(df_part2)  # Automatically rescores with current parameters
    vocab.rescore(scoring_method='enrichment', top_percent=5.0)  # Change to enrichment
    df2 = vocab.to_dataframe()
    print(f'Vocabulary size after enrichment: {len(df2)}')
    print('Top 5 fragments (enrichment scoring):')
    print(df2.head())
    
    # Step 3: Add next 300 lines
    print('\n--- Step 3: Add 300 more molecules ---')
    vocab.add(df_part3)  # Automatically rescores with current parameters
    df3 = vocab.to_dataframe()
    print(f'Final vocabulary size: {len(df3)}')
    print('Top 5 fragments (final enrichment):')
    print(df3.head())
    
    # Verify results
    assert not df1.empty, 'Initial vocabulary should not be empty'
    assert not df2.empty, 'Enrichment vocabulary should not be empty'
    assert not df3.empty, 'Final vocabulary should not be empty'
    assert len(df3) >= len(df2), 'Vocabulary should grow or stay the same'
    
    print('✓ Test 3 passed: Incremental updates with enrichment scoring works')

def test_get_params():
    """Test 4: Get parameters"""
    print('\n' + '='*60)
    print('Test 4: Get parameters')
    print('='*60)
    
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    df_test = pd.read_csv(test_csv, nrows=500)
    print(f'Loaded {len(df_test)} molecules for testing')
    
    # Test average scoring
    vocab = FragmentVocabulary(df_test, min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    print('Initial parameters:')
    print(vocab.get_params())
    
    # Test enrichment scoring
    vocab.rescore(scoring_method='enrichment', top_percent=5.0)
    print('Parameters after enrichment:')
    print(vocab.get_params())

if __name__ == '__main__':
    test_basic_average()
    test_basic_enrichment()
    test_incremental_enrichment()
    test_get_params()
    print('\n' + '='*60)
    print('All tests passed!')
    print('='*60) 