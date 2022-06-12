from speclet.utils import general as gen_utils


def test_merge_sets() -> None:
    sets = [set("a"), set("b"), {"a", "b", "c"}]
    merged_set = gen_utils.merge_sets(sets)
    assert merged_set == {"a", "b", "c"}
