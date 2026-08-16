//! # ZRTL Vector Plugin
//!
//! Provides vector search and embedding operations for machine learning workloads.
//! Essential for RAG (Retrieval-Augmented Generation) pipelines and semantic search.
//!
//! ## Features
//!
//! - Similarity functions (cosine, dot product, Euclidean)
//! - Top-K search with various distance metrics
//! - HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbors
//! - Flat index for exact brute-force search
//! - Embedding normalization and pooling
//!
//! ## Example Use Case
//!
//! ```text
//! // Build a vector search index
//! let index = hnsw_create(384, 10000, 16, 200);
//! hnsw_add_batch(index, ids, embeddings);
//!
//! // Search for similar vectors
//! let results = hnsw_search(index, query_embedding, 10, 50);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use zrtl::zrtl_plugin;

// Import SIMD functions for high-performance vector operations
use zrtl_simd_kernels::{
    vec_cosine_similarity_f32, vec_dot_product_f32, vec_euclidean_f32, vec_euclidean_sq_f32,
    vec_l2_normalize_f32, vec_manhattan_f32,
};

// ============================================================================
// Similarity Functions
// ============================================================================

/// Compute cosine similarity between two vectors using SIMD
/// Returns value in [-1, 1], where 1 means identical direction
#[no_mangle]
pub extern "C" fn vector_cosine_similarity(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    // Use fully optimized SIMD cosine similarity
    vec_cosine_similarity_f32(a, b, len as u64)
}

/// Compute dot product between two vectors using SIMD
#[no_mangle]
pub extern "C" fn vector_dot_product(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    // Use SIMD-accelerated dot product
    vec_dot_product_f32(a, b, len as u64)
}

/// Compute Euclidean distance between two vectors using SIMD
#[no_mangle]
pub extern "C" fn vector_euclidean_distance(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    // Use SIMD-accelerated Euclidean distance
    vec_euclidean_f32(a, b, len as u64)
}

/// Compute squared Euclidean distance (faster, avoids sqrt) using SIMD
#[no_mangle]
pub extern "C" fn vector_euclidean_distance_sq(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    // Use SIMD-accelerated squared Euclidean distance
    vec_euclidean_sq_f32(a, b, len as u64)
}

/// Compute Manhattan (L1) distance between two vectors using SIMD
#[no_mangle]
pub extern "C" fn vector_manhattan_distance(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    // Use SIMD-accelerated Manhattan distance
    vec_manhattan_f32(a, b, len as u64)
}

// ============================================================================
// Batch Similarity Functions
// ============================================================================

/// Compute cosine similarity between query and batch of vectors
/// Results are written to `out` array
#[no_mangle]
pub extern "C" fn vector_cosine_similarity_batch(
    query: *const f32,
    vectors: *const *const f32,
    out: *mut f32,
    n_vectors: usize,
    dim: usize,
) {
    if query.is_null() || vectors.is_null() || out.is_null() || n_vectors == 0 || dim == 0 {
        return;
    }

    unsafe {
        // Pre-compute query norm
        let mut query_norm = 0.0f32;
        for i in 0..dim {
            let q = *query.add(i);
            query_norm += q * q;
        }
        query_norm = query_norm.sqrt();

        if query_norm < 1e-10 {
            for i in 0..n_vectors {
                *out.add(i) = 0.0;
            }
            return;
        }

        for v in 0..n_vectors {
            let vec_ptr = *vectors.add(v);
            if vec_ptr.is_null() {
                *out.add(v) = 0.0;
                continue;
            }

            let mut dot = 0.0f32;
            let mut vec_norm = 0.0f32;

            for i in 0..dim {
                let qi = *query.add(i);
                let vi = *vec_ptr.add(i);
                dot += qi * vi;
                vec_norm += vi * vi;
            }

            vec_norm = vec_norm.sqrt();
            let denom = query_norm * vec_norm;
            *out.add(v) = if denom < 1e-10 { 0.0 } else { dot / denom };
        }
    }
}

// ============================================================================
// Normalization Functions
// ============================================================================

/// L2 normalize a vector in place using SIMD
#[no_mangle]
pub extern "C" fn vector_l2_normalize(data: *mut f32, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    // Use fully optimized SIMD L2 normalize
    vec_l2_normalize_f32(data, len as u64);
}

/// L2 normalize batch of vectors in place
#[no_mangle]
pub extern "C" fn vector_l2_normalize_batch(data: *mut f32, n_vectors: usize, dim: usize) {
    if data.is_null() || n_vectors == 0 || dim == 0 {
        return;
    }

    unsafe {
        for v in 0..n_vectors {
            let vec_ptr = data.add(v * dim);
            vector_l2_normalize(vec_ptr, dim);
        }
    }
}

// ============================================================================
// Pooling Functions
// ============================================================================

/// Mean pooling with attention mask
/// `token_embeddings`: [seq_len, dim] flattened
/// `attention_mask`: [seq_len]
/// `out`: [dim]
#[no_mangle]
pub extern "C" fn vector_mean_pooling(
    token_embeddings: *const f32,
    attention_mask: *const f32,
    out: *mut f32,
    seq_len: usize,
    dim: usize,
) {
    if token_embeddings.is_null() || out.is_null() || seq_len == 0 || dim == 0 {
        return;
    }

    unsafe {
        // Initialize output to zero
        for i in 0..dim {
            *out.add(i) = 0.0;
        }

        let mut mask_sum = 0.0f32;

        for s in 0..seq_len {
            let mask = if attention_mask.is_null() {
                1.0
            } else {
                *attention_mask.add(s)
            };

            if mask > 0.0 {
                mask_sum += mask;
                for d in 0..dim {
                    *out.add(d) += *token_embeddings.add(s * dim + d) * mask;
                }
            }
        }

        // Normalize by mask sum
        if mask_sum > 0.0 {
            for d in 0..dim {
                *out.add(d) /= mask_sum;
            }
        }
    }
}

/// CLS token pooling (take first token embedding)
#[no_mangle]
pub extern "C" fn vector_cls_pooling(token_embeddings: *const f32, out: *mut f32, dim: usize) {
    if token_embeddings.is_null() || out.is_null() || dim == 0 {
        return;
    }

    unsafe {
        ptr::copy_nonoverlapping(token_embeddings, out, dim);
    }
}

// ============================================================================
// Top-K Search
// ============================================================================

/// Search result entry
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

/// Perform top-k search using cosine similarity
/// Returns number of results written to `results`
#[no_mangle]
pub extern "C" fn vector_topk_cosine(
    query: *const f32,
    vectors: *const *const f32,
    ids: *const u64,
    results: *mut SearchResult,
    n_vectors: usize,
    dim: usize,
    k: usize,
) -> usize {
    if query.is_null()
        || vectors.is_null()
        || results.is_null()
        || n_vectors == 0
        || dim == 0
        || k == 0
    {
        return 0;
    }

    let k = k.min(n_vectors);

    unsafe {
        // Pre-compute query norm
        let mut query_norm = 0.0f32;
        for i in 0..dim {
            let q = *query.add(i);
            query_norm += q * q;
        }
        query_norm = query_norm.sqrt();

        if query_norm < 1e-10 {
            return 0;
        }

        // Use min-heap to track top-k
        #[derive(Clone, Copy)]
        struct HeapItem {
            score: f32,
            id: u64,
        }

        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for HeapItem {}

        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse ordering for min-heap behavior (we want to keep highest scores)
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

        for v in 0..n_vectors {
            let vec_ptr = *vectors.add(v);
            if vec_ptr.is_null() {
                continue;
            }

            let mut dot = 0.0f32;
            let mut vec_norm = 0.0f32;

            for i in 0..dim {
                let qi = *query.add(i);
                let vi = *vec_ptr.add(i);
                dot += qi * vi;
                vec_norm += vi * vi;
            }

            vec_norm = vec_norm.sqrt();
            let score = if vec_norm < 1e-10 {
                0.0
            } else {
                dot / (query_norm * vec_norm)
            };

            let id = if ids.is_null() { v as u64 } else { *ids.add(v) };

            if heap.len() < k {
                heap.push(HeapItem { score, id });
            } else if let Some(min) = heap.peek() {
                if score > min.score {
                    heap.pop();
                    heap.push(HeapItem { score, id });
                }
            }
        }

        // Extract results in descending order
        let mut result_vec: Vec<HeapItem> = heap.into_iter().collect();
        result_vec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        for (i, item) in result_vec.iter().enumerate() {
            *results.add(i) = SearchResult {
                id: item.id,
                score: item.score,
            };
        }

        result_vec.len()
    }
}

/// Perform top-k search using dot product
#[no_mangle]
pub extern "C" fn vector_topk_dot(
    query: *const f32,
    vectors: *const *const f32,
    ids: *const u64,
    results: *mut SearchResult,
    n_vectors: usize,
    dim: usize,
    k: usize,
) -> usize {
    if query.is_null()
        || vectors.is_null()
        || results.is_null()
        || n_vectors == 0
        || dim == 0
        || k == 0
    {
        return 0;
    }

    let k = k.min(n_vectors);

    unsafe {
        #[derive(Clone, Copy)]
        struct HeapItem {
            score: f32,
            id: u64,
        }

        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for HeapItem {}

        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

        for v in 0..n_vectors {
            let vec_ptr = *vectors.add(v);
            if vec_ptr.is_null() {
                continue;
            }

            let mut dot = 0.0f32;
            for i in 0..dim {
                dot += *query.add(i) * *vec_ptr.add(i);
            }

            let id = if ids.is_null() { v as u64 } else { *ids.add(v) };

            if heap.len() < k {
                heap.push(HeapItem { score: dot, id });
            } else if let Some(min) = heap.peek() {
                if dot > min.score {
                    heap.pop();
                    heap.push(HeapItem { score: dot, id });
                }
            }
        }

        let mut result_vec: Vec<HeapItem> = heap.into_iter().collect();
        result_vec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        for (i, item) in result_vec.iter().enumerate() {
            *results.add(i) = SearchResult {
                id: item.id,
                score: item.score,
            };
        }

        result_vec.len()
    }
}

/// Perform top-k search using Euclidean distance (returns closest, i.e., smallest distance)
#[no_mangle]
pub extern "C" fn vector_topk_euclidean(
    query: *const f32,
    vectors: *const *const f32,
    ids: *const u64,
    results: *mut SearchResult,
    n_vectors: usize,
    dim: usize,
    k: usize,
) -> usize {
    if query.is_null()
        || vectors.is_null()
        || results.is_null()
        || n_vectors == 0
        || dim == 0
        || k == 0
    {
        return 0;
    }

    let k = k.min(n_vectors);

    unsafe {
        #[derive(Clone, Copy)]
        struct HeapItem {
            distance: f32,
            id: u64,
        }

        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.distance == other.distance
            }
        }
        impl Eq for HeapItem {}

        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                // Max-heap behavior (we want to keep smallest distances)
                self.distance
                    .partial_cmp(&other.distance)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

        for v in 0..n_vectors {
            let vec_ptr = *vectors.add(v);
            if vec_ptr.is_null() {
                continue;
            }

            let mut sum = 0.0f32;
            for i in 0..dim {
                let diff = *query.add(i) - *vec_ptr.add(i);
                sum += diff * diff;
            }
            let distance = sum.sqrt();

            let id = if ids.is_null() { v as u64 } else { *ids.add(v) };

            if heap.len() < k {
                heap.push(HeapItem { distance, id });
            } else if let Some(max) = heap.peek() {
                if distance < max.distance {
                    heap.pop();
                    heap.push(HeapItem { distance, id });
                }
            }
        }

        let mut result_vec: Vec<HeapItem> = heap.into_iter().collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        for (i, item) in result_vec.iter().enumerate() {
            *results.add(i) = SearchResult {
                id: item.id,
                score: item.distance, // Note: for Euclidean, lower is better
            };
        }

        result_vec.len()
    }
}

// ============================================================================
// Flat Index (Brute Force)
// ============================================================================

/// Flat index for exact nearest neighbor search
struct FlatIndex {
    dim: usize,
    vectors: Vec<f32>,
    ids: Vec<u64>,
}

impl FlatIndex {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            ids: Vec::new(),
        }
    }

    fn add(&mut self, id: u64, vector: &[f32]) {
        if vector.len() != self.dim {
            return;
        }
        self.vectors.extend_from_slice(vector);
        self.ids.push(id);
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.len() {
            return None;
        }
        let start = idx * self.dim;
        Some(&self.vectors[start..start + self.dim])
    }
}

pub type FlatIndexPtr = *mut FlatIndex;
pub const FLAT_INDEX_NULL: FlatIndexPtr = std::ptr::null_mut();

/// Create a new flat index
#[no_mangle]
pub extern "C" fn flat_create(dim: usize) -> FlatIndexPtr {
    if dim == 0 {
        return FLAT_INDEX_NULL;
    }

    Box::into_raw(Box::new(FlatIndex::new(dim)))
}

/// Free a flat index
#[no_mangle]
pub extern "C" fn flat_free(index: FlatIndexPtr) {
    if !index.is_null() {
        unsafe {
            drop(Box::from_raw(index));
        }
    }
}

/// Add a vector to the flat index
#[no_mangle]
pub extern "C" fn flat_add(index: FlatIndexPtr, id: u64, vector: *const f32) {
    if index.is_null() || vector.is_null() {
        return;
    }

    unsafe {
        let idx = &mut *index;
        let vec_slice = std::slice::from_raw_parts(vector, idx.dim);
        idx.add(id, vec_slice);
    }
}

/// Get number of vectors in index
#[no_mangle]
pub extern "C" fn flat_len(index: FlatIndexPtr) -> usize {
    if index.is_null() {
        return 0;
    }
    unsafe { (*index).len() }
}

/// Search flat index using cosine similarity
#[no_mangle]
pub extern "C" fn flat_search_cosine(
    index: FlatIndexPtr,
    query: *const f32,
    results: *mut SearchResult,
    k: usize,
) -> usize {
    if index.is_null() || query.is_null() || results.is_null() || k == 0 {
        return 0;
    }

    unsafe {
        let idx = &*index;
        let n = idx.len();
        if n == 0 {
            return 0;
        }

        // Build pointers array
        let mut ptrs: Vec<*const f32> = Vec::with_capacity(n);
        for i in 0..n {
            ptrs.push(idx.vectors.as_ptr().add(i * idx.dim));
        }

        vector_topk_cosine(
            query,
            ptrs.as_ptr(),
            idx.ids.as_ptr(),
            results,
            n,
            idx.dim,
            k,
        )
    }
}

// ============================================================================
// HNSW Index (Hierarchical Navigable Small World)
// ============================================================================

/// HNSW graph layer
struct HNSWLayer {
    /// Neighbors for each node: node_id -> [neighbor_ids]
    neighbors: HashMap<u64, Vec<u64>>,
}

impl HNSWLayer {
    fn new() -> Self {
        Self {
            neighbors: HashMap::new(),
        }
    }

    fn add_node(&mut self, id: u64) {
        self.neighbors.entry(id).or_insert_with(Vec::new);
    }

    fn connect(&mut self, from: u64, to: u64, max_neighbors: usize) {
        if let Some(neighbors) = self.neighbors.get_mut(&from) {
            if !neighbors.contains(&to) && neighbors.len() < max_neighbors {
                neighbors.push(to);
            }
        }
    }

    fn get_neighbors(&self, id: u64) -> &[u64] {
        self.neighbors.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// HNSW index for approximate nearest neighbor search
struct HNSWIndex {
    dim: usize,
    m: usize,     // Number of connections per layer
    m_max: usize, // Max connections at layer 0
    ef_construction: usize,
    ml: f64, // Level multiplier (1/ln(M))

    vectors: HashMap<u64, Vec<f32>>,
    layers: Vec<HNSWLayer>,
    entry_point: Option<u64>,
    max_level: usize,
    next_id: AtomicU64,
}

impl HNSWIndex {
    fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();

        Self {
            dim,
            m,
            m_max: m * 2,
            ef_construction,
            ml,
            vectors: HashMap::new(),
            layers: vec![HNSWLayer::new()], // Start with layer 0
            entry_point: None,
            max_level: 0,
            next_id: AtomicU64::new(0),
        }
    }

    fn random_level(&self) -> usize {
        // Simple random level selection using golden ratio for deterministic pseudo-random
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        // Ensure r is in (0, 1) range, never 0 to avoid -ln(0) = infinity
        let r: f64 = ((id as f64 + 1.0) * 0.618033988749895) % 1.0;
        let r = if r < 1e-10 { 0.5 } else { r }; // Guard against very small values

        // Limit max level to avoid excessive layer creation
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(16) // Cap at reasonable max level
    }

    fn distance(&self, id1: u64, id2: u64) -> f32 {
        let v1 = match self.vectors.get(&id1) {
            Some(v) => v,
            None => return f32::MAX,
        };
        let v2 = match self.vectors.get(&id2) {
            Some(v) => v,
            None => return f32::MAX,
        };

        // Using negative cosine similarity as distance (higher similarity = lower distance)
        let mut dot = 0.0f32;
        let mut norm1 = 0.0f32;
        let mut norm2 = 0.0f32;

        for i in 0..self.dim {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        let denom = (norm1 * norm2).sqrt();
        if denom < 1e-10 {
            f32::MAX
        } else {
            1.0 - (dot / denom) // Convert to distance (0 = identical, 2 = opposite)
        }
    }

    fn distance_to_query(&self, id: u64, query: &[f32]) -> f32 {
        let v = match self.vectors.get(&id) {
            Some(v) => v,
            None => return f32::MAX,
        };

        let mut dot = 0.0f32;
        let mut norm_v = 0.0f32;
        let mut norm_q = 0.0f32;

        for i in 0..self.dim {
            dot += v[i] * query[i];
            norm_v += v[i] * v[i];
            norm_q += query[i] * query[i];
        }

        let denom = (norm_v * norm_q).sqrt();
        if denom < 1e-10 {
            f32::MAX
        } else {
            1.0 - (dot / denom)
        }
    }

    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if vector.len() != self.dim {
            return;
        }

        let level = self.random_level();

        // Ensure we have enough layers
        while self.layers.len() <= level {
            self.layers.push(HNSWLayer::new());
        }

        // Add node to all layers up to its level
        for l in 0..=level {
            self.layers[l].add_node(id);
        }

        // Store vector
        self.vectors.insert(id, vector);

        // If this is the first node, set as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_level = level;
            return;
        }

        let entry = self.entry_point.unwrap();

        // Find entry point at the highest layer
        let mut curr_obj = entry;
        let mut curr_dist = self.distance(id, curr_obj);

        // Traverse from top layer down to insertion level
        for l in (level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &neighbor in self.layers[l].get_neighbors(curr_obj) {
                    let d = self.distance(id, neighbor);
                    if d < curr_dist {
                        curr_dist = d;
                        curr_obj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // Insert at layers from level down to 0
        for l in (0..=level.min(self.max_level)).rev() {
            // Search layer for ef_construction nearest neighbors
            let candidates = self.search_layer(id, curr_obj, self.ef_construction, l);

            // Select M best neighbors and connect
            let m = if l == 0 { self.m_max } else { self.m };
            let neighbors: Vec<u64> = candidates.iter().take(m).map(|&(n, _)| n).collect();

            for &neighbor in &neighbors {
                self.layers[l].connect(id, neighbor, m);
                self.layers[l].connect(neighbor, id, m);
            }

            if !candidates.is_empty() {
                curr_obj = candidates[0].0;
            }
        }

        // Update entry point if new node has higher level
        if level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }
    }

    fn search_layer(&self, query_id: u64, entry: u64, ef: usize, layer: usize) -> Vec<(u64, f32)> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, u64)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(FloatOrd, u64)> = BinaryHeap::new();

        let entry_dist = self.distance(query_id, entry);
        candidates.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
        results.push((FloatOrd(entry_dist), entry));
        visited.insert(entry);

        while let Some(std::cmp::Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
            let worst_dist = results
                .peek()
                .map(|(FloatOrd(d), _)| *d)
                .unwrap_or(f32::MAX);
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            for &neighbor in self.layers[layer].get_neighbors(c_id) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let n_dist = self.distance(query_id, neighbor);
                let worst_dist = results
                    .peek()
                    .map(|(FloatOrd(d), _)| *d)
                    .unwrap_or(f32::MAX);

                if n_dist < worst_dist || results.len() < ef {
                    candidates.push(std::cmp::Reverse((FloatOrd(n_dist), neighbor)));
                    results.push((FloatOrd(n_dist), neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<(u64, f32)> = results
            .into_iter()
            .map(|(FloatOrd(d), id)| (id, d))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }

    fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
        if query.len() != self.dim || self.entry_point.is_none() {
            return Vec::new();
        }

        // Create a temporary ID for the query
        let query_id = u64::MAX; // Use max as sentinel

        // We need to compute distances differently since query isn't stored
        let entry = self.entry_point.unwrap();
        let mut curr_obj = entry;
        let mut curr_dist = self.distance_to_query(curr_obj, query);

        // Traverse from top layer
        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &neighbor in self.layers[l].get_neighbors(curr_obj) {
                    let d = self.distance_to_query(neighbor, query);
                    if d < curr_dist {
                        curr_dist = d;
                        curr_obj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // Search layer 0 with ef_search
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, u64)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(FloatOrd, u64)> = BinaryHeap::new();

        candidates.push(std::cmp::Reverse((FloatOrd(curr_dist), curr_obj)));
        results.push((FloatOrd(curr_dist), curr_obj));
        visited.insert(curr_obj);

        while let Some(std::cmp::Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
            let worst_dist = results
                .peek()
                .map(|(FloatOrd(d), _)| *d)
                .unwrap_or(f32::MAX);
            if c_dist > worst_dist && results.len() >= ef_search {
                break;
            }

            for &neighbor in self.layers[0].get_neighbors(c_id) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let n_dist = self.distance_to_query(neighbor, query);
                let worst_dist = results
                    .peek()
                    .map(|(FloatOrd(d), _)| *d)
                    .unwrap_or(f32::MAX);

                if n_dist < worst_dist || results.len() < ef_search {
                    candidates.push(std::cmp::Reverse((FloatOrd(n_dist), neighbor)));
                    results.push((FloatOrd(n_dist), neighbor));
                    if results.len() > ef_search {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<(u64, f32)> = results
            .into_iter()
            .map(|(FloatOrd(d), id)| (id, d))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec.truncate(k);

        // Convert distance back to similarity
        result_vec.iter().map(|&(id, d)| (id, 1.0 - d)).collect()
    }
}

// Float wrapper for ordering
#[derive(Clone, Copy, PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub type HNSWIndexPtr = *mut HNSWIndex;
pub const HNSW_INDEX_NULL: HNSWIndexPtr = std::ptr::null_mut();

/// Create a new HNSW index
#[no_mangle]
pub extern "C" fn hnsw_create(dim: usize, m: usize, ef_construction: usize) -> HNSWIndexPtr {
    if dim == 0 || m == 0 {
        return HNSW_INDEX_NULL;
    }

    Box::into_raw(Box::new(HNSWIndex::new(dim, m, ef_construction)))
}

/// Free an HNSW index
#[no_mangle]
pub extern "C" fn hnsw_free(index: HNSWIndexPtr) {
    if !index.is_null() {
        unsafe {
            drop(Box::from_raw(index));
        }
    }
}

/// Add a vector to the HNSW index
#[no_mangle]
pub extern "C" fn hnsw_add(index: HNSWIndexPtr, id: u64, vector: *const f32) {
    if index.is_null() || vector.is_null() {
        return;
    }

    unsafe {
        let idx = &mut *index;
        let vec_slice = std::slice::from_raw_parts(vector, idx.dim);
        idx.insert(id, vec_slice.to_vec());
    }
}

/// Get number of vectors in HNSW index
#[no_mangle]
pub extern "C" fn hnsw_len(index: HNSWIndexPtr) -> usize {
    if index.is_null() {
        return 0;
    }
    unsafe { (*index).vectors.len() }
}

/// Search HNSW index
/// Returns number of results written
#[no_mangle]
pub extern "C" fn hnsw_search(
    index: HNSWIndexPtr,
    query: *const f32,
    results: *mut SearchResult,
    k: usize,
    ef_search: usize,
) -> usize {
    if index.is_null() || query.is_null() || results.is_null() || k == 0 {
        return 0;
    }

    unsafe {
        let idx = &*index;
        let query_slice = std::slice::from_raw_parts(query, idx.dim);
        let search_results = idx.search(query_slice, k, ef_search.max(k));

        for (i, (id, score)) in search_results.iter().enumerate() {
            *results.add(i) = SearchResult {
                id: *id,
                score: *score,
            };
        }

        search_results.len()
    }
}

// ============================================================================
// Plugin Registration
// ============================================================================

zrtl_plugin! {
    name: "vector",
    symbols: [
        // Similarity functions
        ("$Vector$cosine_similarity", vector_cosine_similarity),
        ("$Vector$dot_product", vector_dot_product),
        ("$Vector$euclidean_distance", vector_euclidean_distance),
        ("$Vector$euclidean_distance_sq", vector_euclidean_distance_sq),
        ("$Vector$manhattan_distance", vector_manhattan_distance),
        ("$Vector$cosine_similarity_batch", vector_cosine_similarity_batch),

        // Normalization
        ("$Vector$l2_normalize", vector_l2_normalize),
        ("$Vector$l2_normalize_batch", vector_l2_normalize_batch),

        // Pooling
        ("$Vector$mean_pooling", vector_mean_pooling),
        ("$Vector$cls_pooling", vector_cls_pooling),

        // Top-K search
        ("$Vector$topk_cosine", vector_topk_cosine),
        ("$Vector$topk_dot", vector_topk_dot),
        ("$Vector$topk_euclidean", vector_topk_euclidean),

        // Flat index
        ("$Vector$flat_create", flat_create),
        ("$Vector$flat_free", flat_free),
        ("$Vector$flat_add", flat_add),
        ("$Vector$flat_len", flat_len),
        ("$Vector$flat_search_cosine", flat_search_cosine),

        // HNSW index
        ("$Vector$hnsw_create", hnsw_create),
        ("$Vector$hnsw_free", hnsw_free),
        ("$Vector$hnsw_add", hnsw_add),
        ("$Vector$hnsw_len", hnsw_len),
        ("$Vector$hnsw_search", hnsw_search),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let sim = vector_cosine_similarity(a.as_ptr(), b.as_ptr(), 3);
        assert!((sim - 1.0).abs() < 1e-6);

        let c = [0.0f32, 1.0, 0.0];
        let sim2 = vector_cosine_similarity(a.as_ptr(), c.as_ptr(), 3);
        assert!(sim2.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];
        let dist = vector_euclidean_distance(a.as_ptr(), b.as_ptr(), 3);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = [3.0f32, 4.0, 0.0];
        vector_l2_normalize(v.as_mut_ptr(), 3);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_flat_index() {
        let index = flat_create(3);
        assert!(!index.is_null());

        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [0.707f32, 0.707, 0.0];

        flat_add(index, 1, v1.as_ptr());
        flat_add(index, 2, v2.as_ptr());
        flat_add(index, 3, v3.as_ptr());

        assert_eq!(flat_len(index), 3);

        let query = [1.0f32, 0.0, 0.0];
        let mut results = vec![SearchResult { id: 0, score: 0.0 }; 3];

        let n = flat_search_cosine(index, query.as_ptr(), results.as_mut_ptr(), 3);
        assert_eq!(n, 3);
        assert_eq!(results[0].id, 1); // Most similar to query

        flat_free(index);
    }

    #[test]
    fn test_hnsw_index() {
        let index = hnsw_create(3, 4, 10);
        assert!(!index.is_null());

        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [0.707f32, 0.707, 0.0];

        hnsw_add(index, 1, v1.as_ptr());
        hnsw_add(index, 2, v2.as_ptr());
        hnsw_add(index, 3, v3.as_ptr());

        assert_eq!(hnsw_len(index), 3);

        let query = [1.0f32, 0.0, 0.0];
        let mut results = vec![SearchResult { id: 0, score: 0.0 }; 2];

        let n = hnsw_search(index, query.as_ptr(), results.as_mut_ptr(), 2, 10);
        assert!(n > 0);
        assert_eq!(results[0].id, 1); // Most similar to query
        assert!((results[0].score - 1.0).abs() < 0.01); // Cosine similarity ~1.0

        hnsw_free(index);
    }

    #[test]
    fn test_topk_cosine() {
        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [0.707f32, 0.707, 0.0];

        let vectors = [v1.as_ptr(), v2.as_ptr(), v3.as_ptr()];
        let ids = [10u64, 20, 30];
        let query = [1.0f32, 0.0, 0.0];
        let mut results = vec![SearchResult { id: 0, score: 0.0 }; 3];

        let n = vector_topk_cosine(
            query.as_ptr(),
            vectors.as_ptr(),
            ids.as_ptr(),
            results.as_mut_ptr(),
            3,
            3,
            3,
        );

        assert_eq!(n, 3);
        assert_eq!(results[0].id, 10); // Identical
        assert_eq!(results[1].id, 30); // 45 degrees
        assert_eq!(results[2].id, 20); // 90 degrees
    }
}
