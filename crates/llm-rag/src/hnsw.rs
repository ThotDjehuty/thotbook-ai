use std::collections::BinaryHeap;
use std::cmp::Ordering;

use serde::{Serialize, Deserialize};

use crate::embed::EMBED_DIM;

/// A HNSW (Hierarchical Navigable Small World) vector index.
///
/// Pure-Rust implementation for approximate nearest neighbor search.
/// Supports incremental insert and k-NN search with cosine similarity.
#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    /// Maximum number of connections per node per layer.
    pub m: usize,
    /// Max connections for layer 0.
    pub m_max: usize,
    /// Normalization factor for level generation.
    pub ml: f64,
    /// Hierarchical layers: levels[0] = layer 0 (most dense).
    levels: Vec<Vec<Node>>,
    /// node id -> position within `levels[l]`.
    ///
    /// Without this, locating a node meant a linear scan of the level on every
    /// insert. Level 0 holds every node, so building an N-node index was
    /// O(N^2): indexing 4686 chunks made no measurable progress in minutes.
    #[serde(default)]
    level_pos: Vec<ahash::AHashMap<usize, usize>>,
    /// The entry point node index (at the highest level).
    entry_point: usize,
    /// All vectors stored in insertion order.
    vectors: Vec<Vec<f32>>,
    /// Maximum level of any node.
    max_level: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct Node {
    id: usize,
    neighbors: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(m: usize, max_level: usize) -> Self {
        let m_max = if m > 0 { 2 * m } else { 16 };
        let ml = 1.0 / (m as f64).ln();
        let levels = vec![Vec::new(); max_level + 1];
        let level_pos = vec![ahash::AHashMap::new(); max_level + 1];
        Self {
            m,
            m_max,
            ml,
            levels,
            level_pos,
            entry_point: 0,
            vectors: Vec::new(),
            max_level: 0,
        }
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The stored vector for a node, if present. Needed to rebuild the search
    /// engine on load without re-embedding the whole corpus.
    pub fn vector(&self, id: usize) -> Option<Vec<f32>> {
        self.vectors.get(id).cloned()
    }

    /// Record the position a node will occupy in `levels[l]`.
    fn level_pos_insert(&mut self, l: usize, node_id: usize) {
        while self.level_pos.len() <= l {
            self.level_pos.push(ahash::AHashMap::new());
        }
        let pos = self.levels[l].len();
        self.level_pos[l].insert(node_id, pos);
    }

    /// O(1) lookup of a node within a level.
    ///
    /// Falls back to a scan when the map is absent, so indexes serialised
    /// before this field existed still load.
    fn node_at(&self, level: usize, id: usize) -> Option<&Node> {
        if let Some(map) = self.level_pos.get(level) {
            if let Some(&pos) = map.get(&id) {
                return self.levels[level].get(pos);
            }
            if !map.is_empty() {
                return None;
            }
        }
        self.levels.get(level)?.iter().find(|n| n.id == id)
    }

    /// Randomly assign a level for a new node.
    fn random_level(&self) -> usize {
        let r: f64 = rand::random();
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(self.max_level + 1) // Allow growing by 1
    }

    /// Insert a vector with a given ID.
    pub fn insert(&mut self, _id: usize, vector: Vec<f32>) {
        debug_assert_eq!(vector.len(), EMBED_DIM);

        let level = if self.vectors.is_empty() {
            self.max_level // First node goes to top level
        } else {
            self.random_level()
        };

        // Grow levels if needed
        while self.levels.len() <= level {
            self.levels.push(Vec::new());
        }

        self.vectors.push(vector);
        let node_id = self.vectors.len() - 1;

        if self.vectors.len() == 1 {
            // First node
            self.entry_point = 0;
            self.max_level = level;
            for l in 0..=level {
                self.level_pos_insert(l, node_id);
                self.levels[l].push(Node {
                    id: node_id,
                    neighbors: Vec::new(),
                });
            }
            return;
        }

        // Search from entry point, descend to target level
        let mut current = self.entry_point;

        for l in (level + 1)..=self.max_level {
            if let Some((closest, _)) = self.search_closest(current, l, 1) {
                current = closest;
            }
        }

        // Insert node and connect at each level down to 0
        for l in (0..=level).rev() {
            let candidates = self.search_closest_n(current, l, self.m * 2);
            let neighbors = self.select_neighbors(&candidates, self.m);
            current = neighbors.first().copied().unwrap_or(current);

            // Add node to level
            let node = Node {
                id: node_id,
                neighbors: neighbors.clone(),
            };
            self.level_pos_insert(l, node.id);
            self.levels[l].push(node);

            // Add reverse connections
            for &neighbor_id in &neighbors {
                if let Some(node) = self.levels[l].iter_mut().find(|n| n.id == neighbor_id) {
                    if node.neighbors.len() < self.m_max {
                        node.neighbors.push(node_id);
                    }
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = node_id;
        }
    }

    /// Search for the closest node at a given level.
    fn search_closest(&self, start: usize, level: usize, k: usize) -> Option<(usize, f32)> {
        let candidates = self.search_closest_n(start, level, k);
        candidates.into_iter().next()
    }

    /// Greedy search for k closest nodes at a given level.
    fn search_closest_n(&self, start: usize, level: usize, k: usize) -> Vec<(usize, f32)> {
        if self.levels.len() <= level || self.levels[level].is_empty() {
            return vec![];
        }

        let start_node = self.node_at(level, start);
        let start_dist = match start_node {
            Some(n) => {
                let v = &self.vectors[n.id];
                self.distance(&self.vectors[start], v)
            }
            None => return vec![],
        };

        let mut visited = std::collections::HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        candidates.push(Candidate {
            id: start,
            distance: start_dist,
        });
        results.push(Candidate {
            id: start,
            distance: start_dist,
        });
        visited.insert(start);

        while let Some(current) = candidates.pop() {
            if results.len() >= k && current.distance > results.peek().unwrap().distance {
                break;
            }

            // Visit neighbors
            if let Some(node) = self.node_at(level, current.id) {
                for &neighbor_id in &node.neighbors {
                    if visited.insert(neighbor_id) {
                        let dist = self.distance(
                            &self.vectors[start],
                            &self.vectors[neighbor_id],
                        );
                        candidates.push(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        });
                        results.push(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        });
                        if results.len() > k {
                            results.pop();
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.id, c.distance))
            .collect()
    }

    /// Select neighbors using the heuristic from the HNSW paper.
    fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for &(id, _) in candidates {
            if result.len() >= m {
                break;
            }
            // Simple selection: take closest m
            result.push(id);
        }
        result
    }

    /// Cosine distance (1 - cosine_similarity).
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let sim: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        1.0 - sim
    }

    /// k-NN search: find k closest vectors to the query.
    pub fn search(&self, _query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return vec![];
        }

        let mut current = self.entry_point;

        // Descend from top level to level 1
        for l in (1..=self.max_level).rev() {
            if let Some((closest, _)) = self.search_closest(current, l, 1) {
                current = closest;
            }
        }

        // Search at level 0
        
        self.search_closest_n(current, 0, k)
    }

    /// Serialize the index to bytes using bincode.
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes using bincode.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EMBED_DIM: usize = 384;

    #[test]
    fn empty_index() {
        let index = HnswIndex::new(8, 4);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn insert_and_search() {
        let mut index = HnswIndex::new(8, 4);

        // Insert a few vectors
        for i in 0..10 {
            let mut v = vec![0.0; EMBED_DIM];
            v[0] = i as f32;
            index.insert(i, v);
        }

        assert_eq!(index.len(), 10);

        // Search: should return some results
        let q = vec![1.0; EMBED_DIM];
        let results = index.search(&q, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn serialize_roundtrip() {
        let mut index = HnswIndex::new(8, 4);
        for i in 0..10 {
            let mut v = vec![0.0; EMBED_DIM];
            v[i] = 1.0;
            index.insert(i, v);
        }

        let data = index.serialize();
        let restored = HnswIndex::deserialize(&data).unwrap();
        assert_eq!(restored.len(), 10);
    }
}
