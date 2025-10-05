import pandas as pd

def test_no_target_in_features():
    df = pd.read_csv('data/processed/train.csv')
    assert 'Class' in df.columns
    features = [c for c in df.columns if c not in ['Class']]
    assert 'Class' not in features
