//! Undirected graph decompositions.
//!
//! One component so far: the split of a graph into its **biconnected blocks** — the maximal
//! subgraphs no single vertex can disconnect. Hopcroft and Tarjan, *Algorithm 447: Efficient
//! Algorithms for Graph Manipulation*, CACM 16(6), 1973. The recursion of the paper is written
//! here as an explicit walk, because the graphs it runs on are author-drawn and nothing bounds how
//! long a chain of them gets.

/// Split `edges` into biconnected blocks, answering the edge indices of each.
///
/// A block is a maximal set of edges any two of which lie on a common cycle. Every edge belongs to
/// exactly one block; a vertex may belong to several, and a vertex that does is a cut vertex. A
/// tree is the degenerate case: every edge is a block of its own.
///
/// Self-loops and edges naming a vertex at or past `vertex_count` are dropped rather than
/// reported. Isolated vertices produce no block, so an edgeless graph answers empty.
///
/// Runs in time linear in the graph. See the module docs for the citation.
/// The edges at each vertex, by index, with self-loops and out-of-range ends dropped.
fn incidence(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
    for (index, &(from, to)) in edges.iter().enumerate() {
        // Both ends must be in range or neither is listed — half an edge would walk off the end.
        if from == to || from >= vertex_count || to >= vertex_count {
            continue;
        }
        for end in [from, to] {
            if let Some(list) = incident.get_mut(end) {
                list.push(index);
            }
        }
    }
    incident
}

#[must_use]
pub fn biconnected_blocks(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let incident = incidence(vertex_count, edges);
    // An edge only ever entered `incident` with both ends in range, so a walk that reads it back
    // finds both — but the answer stays an Option so no read here can panic.
    let across = |edge: usize, here: usize| {
        edges
            .get(edge)
            .map(|&(from, to)| if from == here { to } else { from })
    };
    let mut discovered: Vec<Option<usize>> = vec![None; vertex_count];
    let mut lowest: Vec<usize> = vec![0; vertex_count];
    let seen_at = |discovered: &[Option<usize>], vertex: usize| {
        discovered.get(vertex).copied().flatten().unwrap_or(0)
    };
    let mut clock = 0usize;
    // Edges met but not yet closed into a block, in the order they were met.
    let mut open: Vec<usize> = Vec::new();
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    // The walk: each frame is a vertex, the edge it was reached by, and how far its incidence list
    // has been read.
    let mut walk: Vec<(usize, Option<usize>, usize)> = Vec::new();
    for root in 0..vertex_count {
        if discovered.get(root).copied().flatten().is_some() {
            continue;
        }
        if let Some(slot) = discovered.get_mut(root) {
            *slot = Some(clock);
        }
        if let Some(slot) = lowest.get_mut(root) {
            *slot = clock;
        }
        clock = clock.saturating_add(1);
        walk.push((root, None, 0));
        while let Some(&(here, arrived, cursor)) = walk.last() {
            let Some(edge) = incident
                .get(here)
                .and_then(|list| list.get(cursor))
                .copied()
            else {
                walk.pop();
                let Some(&(parent, _, _)) = walk.last() else {
                    continue;
                };
                let reach = lowest.get(here).copied().unwrap_or(0);
                if let Some(slot) = lowest.get_mut(parent) {
                    *slot = (*slot).min(reach);
                }
                // Nothing below `here` reaches above `parent`, so `parent` is a cut vertex and
                // everything met since the edge down to `here` closes into one block.
                if reach >= seen_at(&discovered, parent) {
                    let opened = arrived
                        .and_then(|down| open.iter().rposition(|&met| met == down))
                        .unwrap_or(0);
                    blocks.push(open.split_off(opened));
                }
                continue;
            };
            if let Some(frame) = walk.last_mut() {
                frame.2 = cursor.saturating_add(1);
            }
            // Held off by EDGE, not by vertex, so a pair joined twice reports the cycle it is.
            if arrived == Some(edge) {
                continue;
            }
            let Some(next) = across(edge, here) else {
                continue;
            };
            match discovered.get(next).copied().flatten() {
                None => {
                    open.push(edge);
                    if let Some(slot) = discovered.get_mut(next) {
                        *slot = Some(clock);
                    }
                    if let Some(slot) = lowest.get_mut(next) {
                        *slot = clock;
                    }
                    clock = clock.saturating_add(1);
                    walk.push((next, Some(edge), 0));
                }
                // A back edge, counted once — from the end that meets it looking upward.
                Some(seen) if seen < seen_at(&discovered, here) => {
                    open.push(edge);
                    if let Some(slot) = lowest.get_mut(here) {
                        *slot = (*slot).min(seen);
                    }
                }
                Some(_) => {}
            }
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocks as sorted edge lists, sorted among themselves, so a test can name one answer.
    fn blocks_of(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut found = biconnected_blocks(vertex_count, edges);
        for block in &mut found {
            block.sort_unstable();
        }
        found.sort();
        found
    }

    #[test]
    fn a_tree_makes_every_edge_a_block_of_its_own() {
        let edges = [(0, 1), (1, 2), (1, 3)];
        assert_eq!(blocks_of(4, &edges), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn a_rectangle_is_one_block() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        assert_eq!(blocks_of(4, &edges), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn a_cycle_with_a_tail_splits_at_the_cut_vertex() {
        // A triangle 0-1-2 with a spur hanging off vertex 0.
        let edges = [(0, 1), (1, 2), (2, 0), (0, 3)];
        assert_eq!(blocks_of(4, &edges), vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn two_cycles_meeting_at_a_vertex_are_two_blocks() {
        let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)];
        assert_eq!(blocks_of(5, &edges), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn a_doubled_edge_is_a_cycle() {
        let edges = [(0, 1), (0, 1)];
        assert_eq!(blocks_of(2, &edges), vec![vec![0, 1]]);
    }

    #[test]
    fn a_self_loop_and_an_out_of_range_edge_are_dropped() {
        let edges = [(0, 0), (0, 9), (0, 1)];
        assert_eq!(blocks_of(2, &edges), vec![vec![2]]);
    }

    #[test]
    fn an_edgeless_graph_has_no_blocks() {
        assert!(biconnected_blocks(4, &[]).is_empty());
        assert!(biconnected_blocks(0, &[(0, 1)]).is_empty());
    }

    #[test]
    fn every_edge_lands_in_exactly_one_block() {
        // Two rings sharing an edge, a bridge, then a third ring.
        let edges = [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 0),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 4),
        ];
        let found = biconnected_blocks(7, &edges);
        let mut all: Vec<usize> = found.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..edges.len()).collect::<Vec<_>>());
        assert_eq!(
            blocks_of(7, &edges),
            vec![vec![0, 1, 2, 3, 4], vec![5], vec![6, 7, 8]]
        );
    }

    /// A long chain must not blow the stack the way the paper's recursion would.
    #[test]
    fn a_very_long_chain_is_walked_without_recursion() {
        let edges: Vec<(usize, usize)> = (0usize..200_000)
            .map(|at| (at, at.saturating_add(1)))
            .collect();
        assert_eq!(biconnected_blocks(200_001, &edges).len(), edges.len());
    }
}
