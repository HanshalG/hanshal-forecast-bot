from __future__ import annotations

import asyncio
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from src.exa_utils import (
    _call_exa_search_and_contents,
    filter_relevant_exa_results,
    generate_historical_exa_queries,
)
from src.utils import call_asknews_async, call_llm, read_prompt

from .types import EvalStrategyConfig, QbafAgentProfile

ROOT_NODE_ID = "root"
SIMILARITY_FALLBACK = "tfidf_cosine"
_TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")


@dataclass(frozen=True)
class QbafNode:
    id: str
    text: str
    base_score: float
    depth: int
    parent_id: str | None = None
    relation_to_parent: Literal["support", "attack"] | None = None


@dataclass(frozen=True)
class QbafEdge:
    child_id: str
    parent_id: str
    relation: Literal["support", "attack"]


@dataclass(frozen=True)
class AgentQbaf:
    agent_id: str
    root_id: str
    nodes: dict[str, QbafNode]
    edges: list[QbafEdge]


@dataclass(frozen=True)
class QbafNodeInstance:
    id: str
    text: str
    base_score: float
    depth: int
    parent_instance_id: str | None
    relation_to_parent: Literal["support", "attack"] | None


@dataclass(frozen=True)
class CombinedQbafNode:
    id: str
    text: str
    base_score: float
    depth: int
    member_instance_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CombinedQbaf:
    root_id: str
    nodes: dict[str, CombinedQbafNode]
    edges: list[QbafEdge]


@dataclass(frozen=True)
class QbafRunDetails:
    agent_count: int
    combined_node_count: int
    probability_yes: float


def _clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = 0.5
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 14].rstrip() + "\n...[truncated]"


def _parse_json_with_repair(raw: str) -> Any:
    if not isinstance(raw, str):
        return None

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    candidates: list[str] = [cleaned] if cleaned else []
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        candidates.append(m.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

        # Minor repair path for single quotes/trailing commas.
        try:
            repaired = candidate.replace("\t", " ").replace("\r", "")
            repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
            repaired = repaired.replace("'", '"')
            return json.loads(repaired)
        except Exception:
            continue
    return None


def _render_prompt(template: str, values: dict[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize_space(text).lower())


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, av in a.items():
        bv = b.get(key)
        if bv is not None:
            dot += av * bv
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    docs = [_tokenize(text_a), _tokenize(text_b)]
    vocab = sorted(set(docs[0]) | set(docs[1]))
    if not vocab:
        return 0.0
    n_docs = 2
    df: dict[str, int] = {}
    for token in vocab:
        df[token] = int(token in docs[0]) + int(token in docs[1])
    vectors: list[dict[str, float]] = []
    for tokens in docs:
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0
        vec: dict[str, float] = {}
        for token, t in tf.items():
            idf = math.log((n_docs + 1.0) / (df[token] + 1.0)) + 1.0
            vec[token] = t * idf
        vectors.append(vec)
    return _cosine(vectors[0], vectors[1])


def embedding_cosine_similarity(text_a: str, text_b: str, *, dim: int = 256) -> float:
    def _embed(text: str) -> dict[str, float]:
        vec = [0.0] * dim
        for token in _tokenize(text):
            idx = hash(token) % dim
            sign = -1.0 if (hash(token + "_sign") % 2) else 1.0
            vec[idx] += sign
        return {str(i): v for i, v in enumerate(vec) if v != 0.0}

    return _cosine(_embed(text_a), _embed(text_b))


def dfquad_f(strengths: list[float]) -> float:
    if not strengths:
        return 0.0
    acc = _clamp01(strengths[0])
    for strength in strengths[1:]:
        s = _clamp01(strength)
        acc = acc + (1.0 - acc) * s
    return _clamp01(acc)


def dfquad_c(v0: float, va: float, vs: float) -> float:
    v0c = _clamp01(v0)
    vac = _clamp01(va)
    vsc = _clamp01(vs)
    delta = abs(vsc - vac)
    if vac >= vsc:
        return _clamp01(v0c - v0c * delta)
    return _clamp01(v0c + (1.0 - v0c) * delta)


def compute_root_strength(combined_qbaf: CombinedQbaf) -> float:
    if combined_qbaf.root_id not in combined_qbaf.nodes:
        raise ValueError("Combined QBAF has no root node.")

    supporters_by_parent: dict[str, list[str]] = {}
    attackers_by_parent: dict[str, list[str]] = {}
    for edge in combined_qbaf.edges:
        if edge.relation == "support":
            supporters_by_parent.setdefault(edge.parent_id, []).append(edge.child_id)
        else:
            attackers_by_parent.setdefault(edge.parent_id, []).append(edge.child_id)

    nodes_by_depth: dict[int, list[str]] = {}
    for node_id, node in combined_qbaf.nodes.items():
        nodes_by_depth.setdefault(node.depth, []).append(node_id)

    strengths: dict[str, float] = {}
    for depth in sorted(nodes_by_depth.keys(), reverse=True):
        for node_id in nodes_by_depth[depth]:
            node = combined_qbaf.nodes[node_id]
            attacker_strengths = [strengths[ch] for ch in attackers_by_parent.get(node_id, []) if ch in strengths]
            supporter_strengths = [strengths[ch] for ch in supporters_by_parent.get(node_id, []) if ch in strengths]
            va = dfquad_f(attacker_strengths)
            vs = dfquad_f(supporter_strengths)
            strengths[node_id] = dfquad_c(node.base_score, va, vs)

    if combined_qbaf.root_id not in strengths:
        raise ValueError("Could not compute root strength.")
    return _clamp01(strengths[combined_qbaf.root_id])


def _normalize_similarity_key(text: str) -> str:
    return _normalize_space(text).lower()


async def _llm_pairwise_similarity(
    *,
    text_a: str,
    text_b: str,
    model: str,
    similarity_prompt: str,
) -> float:
    prompt = _render_prompt(
        similarity_prompt,
        {
            "text_a": text_a,
            "text_b": text_b,
        },
    )
    response = await call_llm(
        prompt=prompt,
        model=model,
        temperature=0.0,
        reasoning_effort="low",
        component="qbaf_similarity_judge",
    )
    parsed = _parse_json_with_repair(response)
    if isinstance(parsed, dict):
        score = parsed.get("score")
        if isinstance(score, (float, int, str)):
            return _clamp01(score)
    numeric = re.search(r"([01](?:\.\d+)?)", str(response))
    if numeric:
        return _clamp01(numeric.group(1))
    return tfidf_cosine_similarity(text_a, text_b)


async def _similarity_score(
    *,
    text_a: str,
    text_b: str,
    backend: Literal["llm_pairwise", "embedding_cosine", "tfidf_cosine"],
    threshold_cache: dict[str, float],
    llm_model: str,
    similarity_prompt: str,
) -> float:
    left = _normalize_similarity_key(text_a)
    right = _normalize_similarity_key(text_b)
    ordered = tuple(sorted((left, right)))
    key = f"{backend}|{llm_model}|{ordered[0]}|{ordered[1]}"
    if key in threshold_cache:
        return threshold_cache[key]

    if backend == "tfidf_cosine":
        score = tfidf_cosine_similarity(text_a, text_b)
    elif backend == "embedding_cosine":
        score = embedding_cosine_similarity(text_a, text_b)
    else:
        score = await _llm_pairwise_similarity(
            text_a=text_a,
            text_b=text_b,
            model=llm_model,
            similarity_prompt=similarity_prompt,
        )
    threshold_cache[key] = score
    return score


class _Dsu:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


async def _cluster_instance_group(
    *,
    candidate_ids: list[str],
    instances: dict[str, QbafNodeInstance],
    similarity_backend: Literal["llm_pairwise", "embedding_cosine", "tfidf_cosine"],
    similarity_threshold: float,
    similarity_cache: dict[str, float],
    pairwise_model: str,
    similarity_prompt: str,
) -> list[list[str]]:
    if not candidate_ids:
        return []
    if len(candidate_ids) == 1:
        return [[candidate_ids[0]]]

    dsu = _Dsu(candidate_ids)
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(candidate_ids):
        for right in candidate_ids[i + 1 :]:
            pairs.append((left, right))

    scores = await asyncio.gather(
        *[
            _similarity_score(
                text_a=instances[left].text,
                text_b=instances[right].text,
                backend=similarity_backend,
                threshold_cache=similarity_cache,
                llm_model=pairwise_model,
                similarity_prompt=similarity_prompt,
            )
            for left, right in pairs
        ]
    )
    for (left, right), score in zip(pairs, scores):
        if score >= similarity_threshold:
            dsu.union(left, right)

    clusters_by_root: dict[str, list[str]] = {}
    for cid in candidate_ids:
        root = dsu.find(cid)
        clusters_by_root.setdefault(root, []).append(cid)
    return list(clusters_by_root.values())


def _aggregate_base(scores: list[float], mode: Literal["avg", "max"]) -> float:
    if not scores:
        return 0.5
    if mode == "max":
        return _clamp01(max(scores))
    return _clamp01(sum(scores) / len(scores))


def _choose_cluster_text(member_texts: list[str]) -> str:
    if not member_texts:
        return ""
    return sorted(member_texts, key=lambda s: (-len(s), s))[0]


async def combine_agent_qbafs(
    *,
    agent_qbafs: list[AgentQbaf],
    similarity_backend: Literal["llm_pairwise", "embedding_cosine", "tfidf_cosine"],
    similarity_threshold: float,
    base_aggregation: Literal["avg", "max"],
    pairwise_model: str,
) -> CombinedQbaf:
    if not agent_qbafs:
        raise ValueError("No agent QBAFs to combine.")

    similarity_prompt = read_prompt("eval_qbaf_similarity_judge.txt")
    similarity_cache: dict[str, float] = {}

    instances: dict[str, QbafNodeInstance] = {}
    root_instances: list[str] = []
    max_depth = 0
    for qbaf in agent_qbafs:
        for node in qbaf.nodes.values():
            instance_id = f"{qbaf.agent_id}:{node.id}"
            parent_instance_id = f"{qbaf.agent_id}:{node.parent_id}" if node.parent_id else None
            instances[instance_id] = QbafNodeInstance(
                id=instance_id,
                text=node.text,
                base_score=node.base_score,
                depth=node.depth,
                parent_instance_id=parent_instance_id,
                relation_to_parent=node.relation_to_parent,
            )
            if node.depth == 0:
                root_instances.append(instance_id)
            max_depth = max(max_depth, node.depth)

    if not root_instances:
        raise ValueError("Agent QBAFs did not include any root nodes.")

    root_cluster_id = "cluster_root"
    root_base = _aggregate_base([instances[i].base_score for i in root_instances], base_aggregation)
    root_text = _choose_cluster_text([instances[i].text for i in root_instances])
    combined_nodes: dict[str, CombinedQbafNode] = {
        root_cluster_id: CombinedQbafNode(
            id=root_cluster_id,
            text=root_text,
            base_score=root_base,
            depth=0,
            member_instance_ids=sorted(root_instances),
        )
    }

    instance_to_cluster: dict[str, str] = {}
    for root_instance_id in root_instances:
        instance_to_cluster[root_instance_id] = root_cluster_id

    combined_edges: list[QbafEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    next_cluster_idx = 0

    for depth in range(1, max_depth + 1):
        grouped_candidates: dict[tuple[str, str], list[str]] = {}
        for instance in instances.values():
            if instance.depth != depth:
                continue
            if not instance.parent_instance_id or not instance.relation_to_parent:
                continue
            parent_cluster = instance_to_cluster.get(instance.parent_instance_id)
            if not parent_cluster:
                continue
            group_key = (parent_cluster, instance.relation_to_parent)
            grouped_candidates.setdefault(group_key, []).append(instance.id)

        for (parent_cluster, relation), candidate_ids in grouped_candidates.items():
            clusters = await _cluster_instance_group(
                candidate_ids=candidate_ids,
                instances=instances,
                similarity_backend=similarity_backend,
                similarity_threshold=similarity_threshold,
                similarity_cache=similarity_cache,
                pairwise_model=pairwise_model,
                similarity_prompt=similarity_prompt,
            )
            for member_ids in clusters:
                next_cluster_idx += 1
                cluster_id = f"cluster_d{depth}_{next_cluster_idx}"
                base_score = _aggregate_base([instances[mid].base_score for mid in member_ids], base_aggregation)
                combined_nodes[cluster_id] = CombinedQbafNode(
                    id=cluster_id,
                    text=_choose_cluster_text([instances[mid].text for mid in member_ids]),
                    base_score=base_score,
                    depth=depth,
                    member_instance_ids=sorted(member_ids),
                )
                for mid in member_ids:
                    instance_to_cluster[mid] = cluster_id
                edge_key = (cluster_id, parent_cluster, relation)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    combined_edges.append(
                        QbafEdge(
                            child_id=cluster_id,
                            parent_id=parent_cluster,
                            relation=relation,  # type: ignore[arg-type]
                        )
                    )

    return CombinedQbaf(root_id=root_cluster_id, nodes=combined_nodes, edges=combined_edges)


def _extract_argument_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    maybe = parsed.get("arguments")
    if isinstance(maybe, list):
        return [x for x in maybe if isinstance(x, dict)]
    maybe = parsed.get("nodes")
    if isinstance(maybe, list):
        return [x for x in maybe if isinstance(x, dict)]
    return []


def _normalize_relation(value: Any) -> Literal["support", "attack"] | None:
    relation = str(value or "").strip().lower()
    if relation in {"support", "supports", "pro", "for"}:
        return "support"
    if relation in {"attack", "attacks", "con", "against"}:
        return "attack"
    return None


def _normalize_agent_qbaf_from_json(
    *,
    agent_profile: QbafAgentProfile,
    parsed: dict[str, Any],
    question_title: str,
    depth_limit: int,
    max_nodes_per_depth: int,
    root_base_score: float,
) -> AgentQbaf:
    root_raw = parsed.get("root")
    root_text = question_title
    if isinstance(root_raw, dict):
        maybe_root_text = str(root_raw.get("text") or "").strip()
        if maybe_root_text:
            root_text = maybe_root_text

    root = QbafNode(
        id=ROOT_NODE_ID,
        text=_normalize_space(root_text),
        base_score=_clamp01(root_base_score),
        depth=0,
        parent_id=None,
        relation_to_parent=None,
    )

    argument_items = _extract_argument_items(parsed)
    nodes: dict[str, QbafNode] = {root.id: root}
    edges: list[QbafEdge] = []
    accepted_raw_to_new: dict[str, str] = {}
    accepted_raw_depth: dict[str, int] = {}
    accepted_at_depth: dict[int, list[str]] = {}
    per_depth_count: dict[int, int] = {}

    sortable_items: list[tuple[int, int, dict[str, Any]]] = []
    for idx, item in enumerate(argument_items, start=1):
        raw_depth = item.get("depth", 1)
        try:
            parsed_depth = int(raw_depth)
        except Exception:
            parsed_depth = 1
        sortable_items.append((parsed_depth, idx, item))
    sortable_items.sort(key=lambda t: (t[0], t[1]))

    for _, idx, item in sortable_items:
        text = _normalize_space(str(item.get("text") or item.get("argument") or item.get("claim") or ""))
        if not text:
            continue
        relation = _normalize_relation(item.get("relation"))
        if relation is None:
            continue
        try:
            depth = int(item.get("depth", 1))
        except Exception:
            depth = 1
        depth = max(1, min(depth_limit, depth))

        raw_id = str(item.get("id") or item.get("node_id") or f"node_{idx}").strip()
        raw_parent = str(item.get("parent_id") or item.get("parent") or "").strip()
        base_score = _clamp01(item.get("base_score", 0.5))

        parent_id = ROOT_NODE_ID
        effective_depth = depth
        if effective_depth > 1:
            mapped_parent = accepted_raw_to_new.get(raw_parent)
            mapped_depth = accepted_raw_depth.get(raw_parent)
            if mapped_parent is not None and mapped_depth == effective_depth - 1:
                parent_id = mapped_parent
            elif accepted_at_depth.get(effective_depth - 1):
                parent_id = accepted_at_depth[effective_depth - 1][0]
            else:
                parent_id = ROOT_NODE_ID
                effective_depth = 1

        count = per_depth_count.get(effective_depth, 0)
        if count >= max_nodes_per_depth:
            continue
        new_id = f"{agent_profile.id}_d{effective_depth}_{count + 1}"
        per_depth_count[effective_depth] = count + 1
        accepted_raw_to_new[raw_id] = new_id
        accepted_raw_depth[raw_id] = effective_depth
        accepted_at_depth.setdefault(effective_depth, []).append(new_id)

        node = QbafNode(
            id=new_id,
            text=text,
            base_score=base_score,
            depth=effective_depth,
            parent_id=parent_id,
            relation_to_parent=relation,
        )
        nodes[new_id] = node
        edges.append(
            QbafEdge(
                child_id=new_id,
                parent_id=parent_id,
                relation=relation,
            )
        )

    return AgentQbaf(
        agent_id=agent_profile.id,
        root_id=ROOT_NODE_ID,
        nodes=nodes,
        edges=edges,
    )


async def _fetch_exa_context(question_details: dict[str, Any]) -> str:
    try:
        payloads = await generate_historical_exa_queries(question_details)
    except Exception:
        return ""
    if not payloads:
        return ""

    maybe_as_of = str(question_details.get("as_of_time") or "").strip() or None
    chunks: list[str] = []
    for payload in payloads[:3]:
        try:
            results = await asyncio.to_thread(
                _call_exa_search_and_contents,
                payload,
                num_results=6,
                end_published_date_override=maybe_as_of,
            )
        except Exception:
            continue
        if not results:
            continue
        try:
            filtered = await filter_relevant_exa_results(question_details, results)
        except Exception:
            filtered = results
        if not filtered:
            continue

        query = str(payload.get("query") or "").strip()
        if query:
            chunks.append(f"Query: {query}")
        for idx, item in enumerate(filtered[:5], start=1):
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title and not summary:
                continue
            block = f"[{idx}] {title}\n{summary}"
            if url:
                block += f"\nSource: {url}"
            chunks.append(block.strip())
        chunks.append("")
    return _truncate("\n\n".join(chunks).strip(), max_chars=7000)


async def _fetch_agent_evidence(profile: QbafAgentProfile, question_details: dict[str, Any]) -> str:
    if profile.retrieval_mode == "argllm_base":
        return ""
    if profile.retrieval_mode == "rag_asknews":
        try:
            context = await call_asknews_async(question_details)
            return _truncate(str(context), max_chars=12000)
        except Exception:
            return ""
    if profile.retrieval_mode == "rag_exa":
        return await _fetch_exa_context(question_details)
    return ""


async def _estimate_root_base_score(
    *,
    question_details: dict[str, Any],
    evidence_context: str,
    model: str,
) -> float:
    prompt_template = read_prompt("eval_qbaf_root_base_estimate.txt")
    prompt = _render_prompt(
        prompt_template,
        {
            "title": str(question_details.get("title") or ""),
            "description": str(question_details.get("description") or ""),
            "resolution_criteria": str(question_details.get("resolution_criteria") or ""),
            "fine_print": str(question_details.get("fine_print") or ""),
            "evidence_context": evidence_context or "(none)",
        },
    )
    try:
        response = await call_llm(
            prompt=prompt,
            model=model,
            temperature=0.0,
            reasoning_effort="low",
            component="qbaf_root_base_estimate",
        )
    except Exception:
        return 0.5
    parsed = _parse_json_with_repair(response)
    if isinstance(parsed, dict):
        score = parsed.get("root_base_score")
        if isinstance(score, (float, int, str)):
            return _clamp01(score)
    return 0.5


async def _generate_agent_qbaf(
    *,
    strategy: EvalStrategyConfig,
    agent_profile: QbafAgentProfile,
    question_details: dict[str, Any],
    evidence_context: str,
    generation_model: str,
    root_base_score: float,
) -> AgentQbaf:
    prompt_template = read_prompt("eval_qbaf_generate.txt")
    prompt = _render_prompt(
        prompt_template,
        {
            "agent_id": agent_profile.id,
            "agent_description": agent_profile.description or "No additional description.",
            "title": str(question_details.get("title") or ""),
            "description": str(question_details.get("description") or ""),
            "resolution_criteria": str(question_details.get("resolution_criteria") or ""),
            "fine_print": str(question_details.get("fine_print") or ""),
            "as_of_time": str(question_details.get("as_of_time") or ""),
            "evidence_context": evidence_context or "(none)",
            "depth_limit": str(strategy.qbaf_depth),
            "max_nodes_per_depth": str(strategy.qbaf_max_nodes_per_depth),
            "root_base_score": f"{_clamp01(root_base_score):.6f}",
        },
    )
    response = await call_llm(
        prompt=prompt,
        model=generation_model,
        temperature=0.2,
        reasoning_effort="medium",
        component="qbaf_generate",
    )
    parsed = _parse_json_with_repair(response)
    if not isinstance(parsed, dict):
        raise ValueError(f"QBAF generation response was not parseable JSON for agent '{agent_profile.id}'.")

    return _normalize_agent_qbaf_from_json(
        agent_profile=agent_profile,
        parsed=parsed,
        question_title=str(question_details.get("title") or ""),
        depth_limit=strategy.qbaf_depth,
        max_nodes_per_depth=strategy.qbaf_max_nodes_per_depth,
        root_base_score=root_base_score,
    )


def _default_generation_model(strategy: EvalStrategyConfig) -> str:
    return (
        strategy.qbaf_generation_model
        or strategy.model_overrides.get("INSIDE_VIEW_MODEL")
        or os.getenv("INSIDE_VIEW_MODEL")
        or "gpt-5-mini"
    )


def _default_pairwise_model(strategy: EvalStrategyConfig) -> str:
    return (
        strategy.qbaf_pairwise_model
        or strategy.qbaf_generation_model
        or strategy.model_overrides.get("SUMMARY_MODEL")
        or os.getenv("SUMMARY_MODEL")
        or "gpt-5-nano"
    )


async def predict_binary_qbaf(
    *,
    strategy: EvalStrategyConfig,
    question_details: dict[str, Any],
    num_runs: int,
) -> tuple[float, dict[str, Any]]:
    generation_model = _default_generation_model(strategy)
    pairwise_model = _default_pairwise_model(strategy)

    run_probabilities: list[float] = []
    last_error: Exception | None = None
    run_details: list[QbafRunDetails] = []

    for _ in range(max(1, int(num_runs))):
        successful_agents: list[AgentQbaf] = []
        for profile in strategy.qbaf_agent_profiles:
            try:
                evidence_context = await _fetch_agent_evidence(profile, question_details)
                root_base_score = 0.5
                if strategy.qbaf_root_base_mode == "estimated":
                    root_base_score = await _estimate_root_base_score(
                        question_details=question_details,
                        evidence_context=evidence_context,
                        model=generation_model,
                    )
                agent_qbaf = await _generate_agent_qbaf(
                    strategy=strategy,
                    agent_profile=profile,
                    question_details=question_details,
                    evidence_context=evidence_context,
                    generation_model=generation_model,
                    root_base_score=root_base_score,
                )
                successful_agents.append(agent_qbaf)
            except Exception as exc:
                last_error = exc
                continue

        if len(successful_agents) < 2:
            last_error = RuntimeError("QBAF strategy requires at least 2 successful agents.")
            continue

        try:
            backend = strategy.qbaf_similarity_backend
            if backend not in {"llm_pairwise", "embedding_cosine", "tfidf_cosine"}:
                backend = SIMILARITY_FALLBACK
            combined = await combine_agent_qbafs(
                agent_qbafs=successful_agents,
                similarity_backend=backend,  # type: ignore[arg-type]
                similarity_threshold=strategy.qbaf_similarity_threshold,
                base_aggregation=strategy.qbaf_base_aggregation,
                pairwise_model=pairwise_model,
            )
            probability = compute_root_strength(combined)
            probability = min(0.999, max(0.001, probability))
            run_probabilities.append(probability)
            run_details.append(
                QbafRunDetails(
                    agent_count=len(successful_agents),
                    combined_node_count=len(combined.nodes),
                    probability_yes=probability,
                )
            )
        except Exception as exc:
            last_error = exc

    if not run_probabilities:
        if last_error is None:
            raise RuntimeError("QBAF strategy failed to produce a forecast.")
        raise last_error

    median_probability = float(statistics.median(run_probabilities))
    forecast_stddev = float(statistics.pstdev(run_probabilities)) if len(run_probabilities) > 1 else 0.0
    return median_probability, {
        "all_probabilities": [float(x) for x in run_probabilities],
        "forecast_stddev": forecast_stddev,
        "qbaf_run_details": [r.__dict__ for r in run_details],
    }
