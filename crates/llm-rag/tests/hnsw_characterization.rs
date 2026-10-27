//! Characterization tests for `HnswIndex` — they record what the code does
//! TODAY, including a defect, so the defect cannot be fixed silently.
//!
//! `HnswIndex::search(&self, _query: &[f32], k)` discards its query. It
//! descends with `search_closest(current, level, 1)` and ranks with
//! `search_closest_n(start, level, k)`, both of which measure distances
//! BETWEEN STORED VECTORS (`self.vectors[start]` against its neighbours).
//! The query is never compared to anything, so every query returns the same
//! neighbourhood around the graph entry point.
//!
//! Consequence for the paper: Table 3 credits HNSW-only retrieval with
//! Recall@5 = 0.81. A query-independent retriever cannot score that above
//! chance, so the hybrid RRF figure would be carried entirely by BM25.
//!
//! WHEN YOU FIX `search`, `search_is_query_independent` SHOULD FAIL.
//! Replace it with `search_is_query_dependent` (written below, #[ignore]d)
//! which asserts the behaviour you actually want.

use llm_rag::embed::EMBED_DIM;
use llm_rag::hnsw::HnswIndex;

/// A one-hot vector in `EMBED_DIM`, so two different seeds are orthogonal.
fn one_hot(slot: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    v[slot % EMBED_DIM] = 1.0;
    v
}

fn populated(n: usize) -> HnswIndex {
    let mut index = HnswIndex::new(16, 4);
    for i in 0..n {
        index.insert(i, one_hot(i * 7));
    }
    index
}

#[test]
fn search_is_query_independent() {
    let index = populated(64);

    // Two queries that could not be further apart: exact matches for two
    // different stored vectors, orthogonal to each other.
    let a = index.search(&one_hot(0), 5);
    let b = index.search(&one_hot(7 * 31), 5);

    assert!(!a.is_empty(), "index should return something");
    let ids_a: Vec<usize> = a.iter().map(|(id, _)| *id).collect();
    let ids_b: Vec<usize> = b.iter().map(|(id, _)| *id).collect();

    assert_eq!(
        ids_a, ids_b,
        "DEFECT RECORDED: orthogonal queries returned identical results \
         {ids_a:?}. If this assertion now fails, `search` started using its \
         query - delete this test and un-ignore search_is_query_dependent."
    );

    // The zero vector matches nothing, yet still returns the same set.
    let zero = index.search(&vec![0.0f32; EMBED_DIM], 5);
    let ids_zero: Vec<usize> = zero.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids_a, ids_zero, "even the zero query returns the same set");
}

#[test]
fn insert_ignores_the_caller_supplied_id() {
    // `insert(&mut self, _id: usize, vector)` drops `_id`; ids are positional,
    // so a caller numbering documents 100, 200, 300 silently gets 0, 1, 2.
    let mut index = HnswIndex::new(16, 4);
    for (n, id) in [100usize, 200, 300].iter().enumerate() {
        index.insert(*id, one_hot(n * 11));
    }
    assert_eq!(index.len(), 3);
    let hits = index.search(&one_hot(0), 3);
    assert!(
        hits.iter().all(|(id, _)| *id < 3),
        "ids are positional, not the ones supplied: {hits:?}"
    );
}

#[test]
fn search_on_empty_index_is_empty() {
    let index = HnswIndex::new(16, 4);
    assert!(index.search(&one_hot(0), 5).is_empty());
}

#[test]
fn roundtrip_through_serde_preserves_length() {
    let index = populated(16);
    let bytes = index.serialize();
    let back = HnswIndex::deserialize(&bytes).expect("deserialize");
    assert_eq!(back.len(), index.len());
}

/// The test to enable once `search` compares against the query: an exact
/// match for a stored vector must come back first.
#[test]
#[ignore = "enable after HnswIndex::search starts using its query vector"]
fn search_is_query_dependent() {
    let index = populated(64);
    for probe in [0usize, 5, 17, 40] {
        let want = one_hot(probe * 7);
        let hits = index.search(&want, 5);
        assert_eq!(
            hits.first().map(|(id, _)| *id),
            Some(probe),
            "an exact match for stored vector {probe} should rank first, got {hits:?}"
        );
    }
}
