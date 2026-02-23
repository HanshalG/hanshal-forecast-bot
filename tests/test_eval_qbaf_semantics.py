from src.eval.qbaf_strategy import CombinedQbaf, CombinedQbafNode, QbafEdge, compute_root_strength, dfquad_c, dfquad_f


def test_dfquad_f_edge_cases():
    assert dfquad_f([]) == 0.0
    assert dfquad_f([0.3]) == 0.3
    assert dfquad_f([0.6, 0.5]) == 0.8


def test_dfquad_c_edge_cases():
    assert dfquad_c(0.5, 0.0, 0.0) == 0.5
    # stronger attack than support lowers base
    assert dfquad_c(0.8, 0.7, 0.1) < 0.8
    # stronger support than attack raises base
    assert dfquad_c(0.2, 0.1, 0.7) > 0.2


def test_root_strength_range_and_monotonicity():
    root = CombinedQbafNode(id="root", text="claim", base_score=0.5, depth=0)
    weak_support = CombinedQbafNode(id="s1", text="weak support", base_score=0.2, depth=1)
    strong_support = CombinedQbafNode(id="s2", text="strong support", base_score=0.9, depth=1)

    with_weak = CombinedQbaf(
        root_id="root",
        nodes={"root": root, "s1": weak_support},
        edges=[QbafEdge(child_id="s1", parent_id="root", relation="support")],
    )
    with_strong = CombinedQbaf(
        root_id="root",
        nodes={"root": root, "s2": strong_support},
        edges=[QbafEdge(child_id="s2", parent_id="root", relation="support")],
    )

    weak_score = compute_root_strength(with_weak)
    strong_score = compute_root_strength(with_strong)

    assert 0.0 <= weak_score <= 1.0
    assert 0.0 <= strong_score <= 1.0
    assert strong_score > weak_score
