import os
import pandas as pd
from argo.vocab import FragmentVocabulary

def test_fragment_vocabulary():
    test_csv = os.path.join(os.path.dirname(__file__), 'test_vocab.csv')

    # Test average scoring
    vocab = FragmentVocabulary(test_csv, min_frag_size=5, max_frag_size=30, min_count=5, max_fragments=10, verbose=False, lower_is_better=True)
    df = vocab.to_dataframe()
    print('Average scoring:')
    print(df.head())
    assert not df.empty, 'Vocabulary DataFrame should not be empty.'
    assert 'frag' in df.columns and 'score' in df.columns, 'Missing required columns.'

    # Test enrichment scoring
    vocab.craft_vocabulary(scoring_method='enrichment', top_percent=10.0, max_fragments=10)
    df2 = vocab.to_dataframe()
    print('Enrichment scoring:')
    print(df2.head())
    assert not df2.empty, 'Enrichment vocabulary DataFrame should not be empty.'

    # Test DataFrame-like access
    assert len(vocab) == len(df2)
    assert isinstance(vocab.head(3), pd.DataFrame)
    print('All tests passed.')

if __name__ == '__main__':
    test_fragment_vocabulary() 