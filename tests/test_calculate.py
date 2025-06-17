import pandas as pd
from src.calculate import calculate_rs_factor

def test_rs_factor_constant_series():
    df = pd.DataFrame({"close": [100]*252})
    assert calculate_rs_factor(df) == 0