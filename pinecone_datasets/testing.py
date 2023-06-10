from pandas.testing import assert_frame_equal


def assert_datasets_equal(ds1, ds2):
    assert_frame_equal(ds1.documents, ds2.documents)
    assert_frame_equal(ds1.queries, ds2.queries)
    assert ds1.metadata == ds2.metadata
