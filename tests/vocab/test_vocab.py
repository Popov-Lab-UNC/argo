import os
import pandas as pd
import tempfile
from argo.vocab import FragmentVocabulary

def test_basic_average():
    """Test 1: Basic loading with average scoring"""
    print('='*60)
    print('Test 1: Basic loading with average scoring')
    print('='*60)
    
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    
    df_test = pd.read_csv(test_csv, nrows=500)
    print(f'Loaded {len(df_test)} molecules for testing')
    
    vocab = FragmentVocabulary(slicer='f-rag', data=df_test, min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    df = vocab.get_vocab()
    
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
    
    df_test = pd.read_csv(test_csv, nrows=500)
    print(f'Loaded {len(df_test)} molecules for testing')
    
    vocab = FragmentVocabulary(slicer='f-rag', data=df_test, scoring_method='enrichment', top_percent=5.0,
                             min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    df = vocab.get_vocab()
    
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
    
    all_data = pd.read_csv(test_csv)
    df_part1 = all_data.iloc[:300].copy()
    df_part2 = all_data.iloc[300:700].copy()
    df_part3 = all_data.iloc[700:1000].copy()
    
    print(f'Part 1: {len(df_part1)}, Part 2: {len(df_part2)}, Part 3: {len(df_part3)}')
    
    print('\n--- Step 1: Initial load with average scoring ---')
    vocab = FragmentVocabulary(slicer='f-rag', data=df_part1, scoring_method='average',
                             min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=1000, verbose=False)
    df1 = vocab.get_vocab()
    print(f'Initial vocabulary size: {len(df1)}')
    
    print('\n--- Step 2: Add 400 more molecules and switch to enrichment ---')
    vocab.add(df_part2) # Does not rescore
    vocab.rescore(scoring_method='enrichment', top_percent=5.0) # Rescore with all data
    df2 = vocab.get_vocab()
    print(f'Vocabulary size after enrichment: {len(df2)}')
    
    print('\n--- Step 3: Add 300 more molecules and rescore with enrichment ---')
    vocab.add(df_part3, rescore=True) # Adds data and rescores with existing enrichment params
    df3 = vocab.get_vocab()
    print(f'Final vocabulary size: {len(df3)}')
    
    assert not df1.empty
    assert not df2.empty
    assert not df3.empty
    assert len(df3) >= len(df2)
    print('✓ Test 3 passed: Incremental updates with enrichment scoring works')

def test_get_params():
    """Test 4: Get parameters"""
    print('\n' + '='*60)
    print('Test 4: Get parameters')
    print('='*60)
    
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    df_test = pd.read_csv(test_csv, nrows=500)
    
    vocab = FragmentVocabulary(slicer='f-rag', data=df_test, min_frag_size=5, max_frag_size=30, min_count=5, verbose=False)
    print('Initial parameters:', vocab.get_params())
    
    vocab.rescore(scoring_method='enrichment', top_percent=5.0)
    print('Parameters after enrichment:', vocab.get_params())

    assert vocab.get_params()['scoring_method'] == 'enrichment'
    print('✓ Test 4 passed: Get parameters works')

def test_save_load():
    """Test 5: Save and load functionality"""
    print('\n' + '='*60)
    print('Test 5: Save and load functionality')
    print('='*60)

    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')
    df_test = pd.read_csv(test_csv, nrows=100)

    vocab = FragmentVocabulary(slicer='f-rag', data=df_test, scoring_method='average', min_count=2, verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        vocab.save(tmp.name)
        print(f'Vocabulary saved to {tmp.name}')

        loaded_vocab = FragmentVocabulary.load(tmp.name)
        print('Vocabulary loaded successfully')

    assert len(vocab.get_vocab()) == len(loaded_vocab.get_vocab())
    assert vocab.get_params() == loaded_vocab.get_params()
    assert pd.testing.assert_frame_equal(vocab.get_data(), loaded_vocab.get_data()) is None
    print('✓ Test 5 passed: Save and load functionality works')

if __name__ == '__main__':
    test_basic_average()
    test_basic_enrichment()
    test_incremental_enrichment()
    test_get_params()
    test_save_load()
    print('\n' + '='*60)
    print('All tests passed!')
    print('='*60) 