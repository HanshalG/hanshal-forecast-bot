import asyncio

from src.eval.qbaf_strategy import AgentQbaf, QbafEdge, QbafNode, combine_agent_qbafs


def _agent(agent_id: str, nodes: list[QbafNode], edges: list[QbafEdge]) -> AgentQbaf:
    node_map = {n.id: n for n in nodes}
    return AgentQbaf(agent_id=agent_id, root_id="root", nodes=node_map, edges=edges)


def test_combinator_respects_same_parent_and_same_relation():
    a1 = _agent(
        "a1",
        nodes=[
            QbafNode(id="root", text="Claim", base_score=0.5, depth=0),
            QbafNode(id="p1", text="Parent one", base_score=0.6, depth=1, parent_id="root", relation_to_parent="support"),
            QbafNode(id="c1", text="shared child text", base_score=0.7, depth=2, parent_id="p1", relation_to_parent="support"),
        ],
        edges=[
            QbafEdge(child_id="p1", parent_id="root", relation="support"),
            QbafEdge(child_id="c1", parent_id="p1", relation="support"),
        ],
    )
    a2 = _agent(
        "a2",
        nodes=[
            QbafNode(id="root", text="Claim", base_score=0.5, depth=0),
            QbafNode(id="p2", text="Different parent", base_score=0.55, depth=1, parent_id="root", relation_to_parent="support"),
            QbafNode(id="c2", text="shared child text", base_score=0.65, depth=2, parent_id="p2", relation_to_parent="support"),
        ],
        edges=[
            QbafEdge(child_id="p2", parent_id="root", relation="support"),
            QbafEdge(child_id="c2", parent_id="p2", relation="support"),
        ],
    )

    combined = asyncio.run(
        combine_agent_qbafs(
            agent_qbafs=[a1, a2],
            similarity_backend="tfidf_cosine",
            similarity_threshold=0.95,
            base_aggregation="avg",
            pairwise_model="gpt-5-nano",
        )
    )

    depth2_nodes = [n for n in combined.nodes.values() if n.depth == 2]
    assert len(depth2_nodes) == 2


def test_combinator_threshold_behavior():
    a1 = _agent(
        "a1",
        nodes=[
            QbafNode(id="root", text="Claim", base_score=0.5, depth=0),
            QbafNode(id="s1", text="inflation is falling", base_score=0.8, depth=1, parent_id="root", relation_to_parent="support"),
        ],
        edges=[QbafEdge(child_id="s1", parent_id="root", relation="support")],
    )
    a2 = _agent(
        "a2",
        nodes=[
            QbafNode(id="root", text="Claim", base_score=0.5, depth=0),
            QbafNode(id="s2", text="inflation is falling", base_score=0.2, depth=1, parent_id="root", relation_to_parent="support"),
        ],
        edges=[QbafEdge(child_id="s2", parent_id="root", relation="support")],
    )

    merged = asyncio.run(
        combine_agent_qbafs(
            agent_qbafs=[a1, a2],
            similarity_backend="tfidf_cosine",
            similarity_threshold=0.99,
            base_aggregation="avg",
            pairwise_model="gpt-5-nano",
        )
    )
    unmerged = asyncio.run(
        combine_agent_qbafs(
            agent_qbafs=[a1, a2],
            similarity_backend="tfidf_cosine",
            similarity_threshold=1.01,
            base_aggregation="avg",
            pairwise_model="gpt-5-nano",
        )
    )

    assert len([n for n in merged.nodes.values() if n.depth == 1]) == 1
    assert len([n for n in unmerged.nodes.values() if n.depth == 1]) == 2


def test_combinator_base_aggregation_avg_vs_max():
    a1 = _agent(
        "a1",
        nodes=[
            QbafNode(id="root", text="Claim", base_score=0.5, depth=0),
            QbafNode(id="s1", text="same argument", base_score=0.2, depth=1, parent_id="root", relation_to_parent="support"),
        ],
        edges=[QbafEdge(child_id="s1", parent_id="root", relation="support")],
    )
    a2 = _agent(
        "a2",
        nodes=[
            QbafNode(id="root", text="Claim", base_score=0.5, depth=0),
            QbafNode(id="s2", text="same argument", base_score=0.8, depth=1, parent_id="root", relation_to_parent="support"),
        ],
        edges=[QbafEdge(child_id="s2", parent_id="root", relation="support")],
    )

    avg_combined = asyncio.run(
        combine_agent_qbafs(
            agent_qbafs=[a1, a2],
            similarity_backend="tfidf_cosine",
            similarity_threshold=0.99,
            base_aggregation="avg",
            pairwise_model="gpt-5-nano",
        )
    )
    max_combined = asyncio.run(
        combine_agent_qbafs(
            agent_qbafs=[a1, a2],
            similarity_backend="tfidf_cosine",
            similarity_threshold=0.99,
            base_aggregation="max",
            pairwise_model="gpt-5-nano",
        )
    )

    avg_depth1 = [n for n in avg_combined.nodes.values() if n.depth == 1]
    max_depth1 = [n for n in max_combined.nodes.values() if n.depth == 1]
    assert len(avg_depth1) == 1
    assert len(max_depth1) == 1
    assert avg_depth1[0].base_score == 0.5
    assert max_depth1[0].base_score == 0.8
