from __future__ import annotations

from abc import ABC, abstractmethod

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

from .utils import RND_GEN, VALID_TKN, INVALID_TKN

from numpy import (
    ndarray,
    array,

    exp,
)

# Matching Strategy Class

class MatchingStrategy(ABC):
    def __init__(self, name = "MatchingStrategy", deterministic_partner= True) -> None:
        self.name = name
        self.deterministic_partner = deterministic_partner

    @abstractmethod
    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray: ...

    def process_graph(self, graph: TripartiteGraph): 
        pass

    @abstractmethod
    def select_inode_for_L(self, graph: TripartiteGraph, lnode: LNode) -> INode | None: ...

    @abstractmethod
    def select_inode_for_R(self, graph: TripartiteGraph, rnode: RNode) -> INode | None: ...

    def select_partner(self, graph: TripartiteGraph, nodes: set[varNode]):
        if not nodes:
            return None
        return next(iter(sorted(nodes)))

    def reset(self, graph:TripartiteGraph):
        pass

class RandomStrategy(MatchingStrategy):
    def __init__(self, name="RandomStrategy", deterministic_partner=False) -> None:
        super().__init__(name, deterministic_partner)

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode_scores = []

        has_valid = False
        candidate_set = set(node.candidate_Inodes)
        for inode_id in graph.Inodes:
            inode = graph.Inodes[inode_id]
            valid = inode.available and (inode_id in candidate_set)

            if valid:
                inode_scores.append(VALID_TKN)
                has_valid = True
            else:
                inode_scores.append(INVALID_TKN)

        # WAIT action
        inode_scores.append(INVALID_TKN if has_valid else VALID_TKN)

        scores = array(inode_scores)
        return scores

    def _get_random_available_inode(self, graph: TripartiteGraph, node: varNode) -> INode | None:
        available_candidates = []
        
        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                available_candidates.append(inode)

        # if no candidates are free
        if not available_candidates: return None
        idx = RND_GEN.integers(len(available_candidates))
        return available_candidates[idx]

    def select_inode_for_L(self, graph, lnode):
        return self._get_random_available_inode(graph, lnode)

    def select_inode_for_R(self, graph, rnode):
        return self._get_random_available_inode(graph, rnode)

    def select_partner(self, graph, nodes: set[varNode]):
        if(self.deterministic_partner):
            return super().select_partner(graph, nodes)

        if not nodes:
            return None

        node_tuple = tuple(nodes)
        idx = RND_GEN.integers(len(node_tuple))
        return node_tuple[idx]

class GreedyStrategy(MatchingStrategy):
    def __init__(self, name="GreedyStrategy",) -> None:
        super().__init__(name, True)

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode= None

        for inode_id in node.candidate_Inodes:
            inode= graph.Inodes[inode_id]
            if(inode.available): 
                break

        inode_scores = list([INVALID_TKN] * len(graph.Inodes))

        # WAIT action
        if(inode):
            inode_idx = tuple(graph.Inodes.keys()).index(inode.id)

            inode_scores[inode_idx] = VALID_TKN
            inode_scores.append(INVALID_TKN)
        else:
            inode_scores.append(VALID_TKN)

        scores = array(inode_scores)
        return scores

    def select_inode_for_L(self, graph, lnode):
        # Optimal with R connected
        for inode_id in lnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                return inode
        return None

    def select_inode_for_R(self, graph, rnode):
        # Optimal with L connected
        for inode_id in rnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                return inode
        return None

class RankStrategy(MatchingStrategy):
    def __init__(self, name= "RankStrategy", deterministic_partner= False) -> None:
        super().__init__(name, deterministic_partner)

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode_scores = []

        candidate_ids = [
            inode_id for inode_id in node.candidate_Inodes
            if graph.Inodes[inode_id].available
        ]

        # Case 4: no available neighbors → WAIT
        if not candidate_ids:
            scores = [INVALID_TKN for _ in graph.Inodes]
            scores.append(VALID_TKN)
            return array(scores)

        # ---- lo2 (lowest ranked neighbor) ----
        lo2_id = min(candidate_ids, key=lambda i: graph.Inodes[i].rank)
        lo2_rank = graph.Inodes[lo2_id].rank

        # ---- check opposite-side ----
        if node.node_type == 'L':
            lo2_has_opposite = len(graph.right_memory[lo2_id]) > 0
        else:
            lo2_has_opposite = len(graph.left_memory[lo2_id]) > 0

        # ---- compute scores ----
        has_valid= False
        candidate_set = set(node.candidate_Inodes)
        for inode_id in graph.Inodes:
            inode = graph.Inodes[inode_id]

            # Invalid nodes
            if not inode.available or (inode_id not in candidate_set):
                inode_scores.append(INVALID_TKN)
                continue

            if lo2_has_opposite:
                # Case 2: ε = 0, only lo2 is valid
                if inode_id == lo2_id:
                    score = exp(inode.rank - 1)
                    has_valid = True
                else:
                    score = INVALID_TKN
            else:
                # Case 3 or Case 1
                if ((node.node_type == 'L' and graph.right_memory[inode_id]) or 
                    (node.node_type == 'R' and graph.left_memory[inode_id])):
                    # eligible alternative (like lo2.5)
                    score = exp(lo2_rank - 1)
                    has_valid = True
                else:
                    # no opposite edge → cannot match
                    score = INVALID_TKN

            inode_scores.append(score)
        
        # Wait not prefered if has_valid
        if has_valid:
            inode_scores.append(INVALID_TKN)
        else:
            inode_scores.append(VALID_TKN)

        scores = array(inode_scores)
        return scores

    def process_graph(self, graph):
        for inode in graph.Inodes.values():
            inode.rank = RND_GEN.random()

    def select_inode_for_L(self, graph, lnode):
        sorted_ids = sorted(lnode.candidate_Inodes, key=lambda id: graph.Inodes[id].rank)   

        best_available = None
        best_valid = None

        for inode_id in sorted_ids:
            inode = graph.Inodes[inode_id]
            if not inode.available:
                continue

            if best_available is None:
                best_available = inode  # Case 4

            if graph.right_memory[inode_id]:
                best_valid = inode
                break   

        if best_valid: return best_valid   # Case 2 or 3
        return None # Case 1 -> Wait

    def select_inode_for_R(self, graph, rnode):
        sorted_ids = sorted(rnode.candidate_Inodes, key=lambda id: graph.Inodes[id].rank)   

        best_available = None
        best_valid = None

        for inode_id in sorted_ids:
            inode = graph.Inodes[inode_id]
            if not inode.available:
                continue

            if best_available is None:
                best_available = inode  # Case 4

            if graph.left_memory[inode_id]:
                best_valid = inode
                break   

        if best_valid: return best_valid   # Case 2 or 3
        return None # Case 1 -> Wait

    def select_partner(self, graph, nodes):
        if(self.deterministic_partner):
            return super().select_partner(graph, nodes)

        if not nodes: return None

        nodes = tuple(nodes)
        idx = RND_GEN.integers(len(nodes))
        return nodes[idx]


class MinDegreeStrategy(MatchingStrategy):
    def __init__(self, name="MinDegreeStrategy", deterministic_partner=False) -> None:
        super().__init__(name, deterministic_partner)

    def _inode_degree(self, graph: TripartiteGraph, inode_id: int) -> int:
        return len(graph.left_memory[inode_id]) + len(graph.right_memory[inode_id])

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode_scores = []

        candidate_ids = [
            inode_id for inode_id in node.candidate_Inodes
            if graph.Inodes[inode_id].available
        ]

        # No candidates → WAIT
        if not candidate_ids:
            scores = [INVALID_TKN for _ in graph.Inodes]
            scores.append(VALID_TKN)
            return array(scores)

        min_degree_any = float('inf')
        min_degree_valid = float('inf')

        for inode_id in candidate_ids:
            deg = self._inode_degree(graph, inode_id)

            # update global min
            if deg < min_degree_any:
                min_degree_any = deg

            # check opposite validity
            if node.node_type == 'L':
                valid = len(graph.right_memory[inode_id]) > 0
            else:
                valid = len(graph.left_memory[inode_id]) > 0

            if valid and deg < min_degree_valid:
                min_degree_valid = deg

        # ---- Decide which min to use ----
        use_valid = (min_degree_valid < float('inf'))

        candidate_set = set(node.candidate_Inodes)
        for inode_id in graph.Inodes:
            inode = graph.Inodes[inode_id]

            if not inode.available or (inode_id not in candidate_set):
                inode_scores.append(INVALID_TKN)
                continue

            deg = self._inode_degree(graph, inode_id)

            if node.node_type == 'L':
                valid = len(graph.right_memory[inode_id]) > 0
            else:
                valid = len(graph.left_memory[inode_id]) > 0

            if use_valid:
                # Only best valid nodes
                if valid and deg == min_degree_valid:
                    score = VALID_TKN
                else:
                    score = INVALID_TKN
            else:
                # fallback: best overall nodes
                if deg == min_degree_any:
                    score = VALID_TKN
                else:
                    score = INVALID_TKN

            inode_scores.append(score)

        # WAIT discouraged if we have candidates
        inode_scores.append(INVALID_TKN)

        return array(inode_scores)

    def _select_min_degree(self, graph: TripartiteGraph, node: varNode):
        best_valid = None
        best_valid_deg = float('inf')

        best_any = None
        best_any_deg = float('inf')

        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]

            if not inode.available: continue

            deg = self._inode_degree(graph, inode_id)

            # Track best overall (fallback)
            if deg < best_any_deg:
                best_any_deg = deg
                best_any = inode

            # Check opposite-side validity
            if node.node_type == 'L':
                valid = len(graph.right_memory[inode_id]) > 0
            else:
                valid = len(graph.left_memory[inode_id]) > 0

            # Track best valid
            if valid and deg < best_valid_deg:
                best_valid_deg = deg
                best_valid = inode

        # Prefer valid match, else fallback
        if best_valid:
            return best_valid

        return best_any

    def select_inode_for_L(self, graph, lnode):
        inode = self._select_min_degree(graph, lnode)
        if inode:
            return inode

        return None

    def select_inode_for_R(self, graph, rnode):
        inode = self._select_min_degree(graph, rnode)
        if inode:
            return inode

        return None

    def select_partner(self, graph, nodes):
        if self.deterministic_partner:
            return super().select_partner(graph, nodes)

        if not nodes:
            return None

        nodes = tuple(nodes)
        idx = RND_GEN.integers(len(nodes))
        return nodes[idx]

from .GraphModel import TripartiteGraph