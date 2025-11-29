import inspect


def test_classify_and_categorize_default_sorting():
    # Verify the function signature default is sort_results=False so results are not auto-sorted
    import classify_many
    sig = inspect.signature(classify_many.classify_and_categorize)
    param = sig.parameters.get('sort_results')
    assert param is not None, 'classify_and_categorize should accept a sort_results parameter'
    assert param.default is False, 'sort_results should default to False (do not sort results)'
