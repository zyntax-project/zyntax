//! # ZRTL Text Plugin
//!
//! Provides text processing and tokenization for machine learning workloads.
//! Essential for NLP models, LLMs, and text-based ML pipelines.
//!
//! ## Features
//!
//! - BPE tokenization (GPT-2/GPT-3/LLaMA style)
//! - Text preprocessing (lowercase, normalization)
//! - Text chunking for RAG pipelines
//! - Basic text utilities
//!
//! ## Tokenizer Support
//!
//! This plugin provides a simple BPE tokenizer implementation. For production
//! use with models like Whisper, LLaMA, or GPT, load the appropriate vocab
//! and merges files.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use unicode_normalization::UnicodeNormalization;
use zrtl::zrtl_plugin;

// ============================================================================
// Tokenizer Handle
// ============================================================================

/// BPE tokenizer data
struct BPETokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// BPE merge rules: (pair) -> merged_token
    merges: Vec<(String, String)>,
    /// Special tokens
    pad_token: Option<u32>,
    eos_token: Option<u32>,
    bos_token: Option<u32>,
    unk_token: Option<u32>,
}

impl BPETokenizer {
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            pad_token: None,
            eos_token: None,
            bos_token: None,
            unk_token: None,
        }
    }

    fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Encode text to token IDs using BPE
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with character-level tokens
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges greedily
        for (left, right) in &self.merges {
            let merged = format!("{}{}", left, right);
            let mut i = 0;
            while i + 1 < tokens.len() {
                if &tokens[i] == left && &tokens[i + 1] == right {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert tokens to IDs
        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().or(self.unk_token).unwrap_or(0))
            .collect()
    }

    /// Decode token IDs to text
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .cloned()
            .collect()
    }
}

/// Global tokenizer storage
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
static TOKENIZER_STORE: Mutex<Option<HashMap<u64, BPETokenizer>>> = Mutex::new(None);

fn init_store() {
    let mut store = TOKENIZER_STORE.lock().unwrap();
    if store.is_none() {
        *store = Some(HashMap::new());
    }
}

fn insert_tokenizer(tokenizer: BPETokenizer) -> u64 {
    init_store();
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut store = TOKENIZER_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        map.insert(handle, tokenizer);
    }
    handle
}

fn get_tokenizer<F, R>(handle: u64, f: F) -> R
where
    F: FnOnce(Option<&BPETokenizer>) -> R,
{
    init_store();
    let store = TOKENIZER_STORE.lock().unwrap();
    if let Some(ref map) = *store {
        f(map.get(&handle))
    } else {
        f(None)
    }
}

fn take_tokenizer(handle: u64) -> Option<BPETokenizer> {
    init_store();
    let mut store = TOKENIZER_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        map.remove(&handle)
    } else {
        None
    }
}

pub type TokenizerHandle = u64;
pub const TOKENIZER_NULL: TokenizerHandle = 0;

// ============================================================================
// Tokenizer Creation
// ============================================================================

/// Create an empty tokenizer
#[no_mangle]
pub extern "C" fn tokenizer_create() -> TokenizerHandle {
    insert_tokenizer(BPETokenizer::new())
}

/// Free tokenizer
#[no_mangle]
pub extern "C" fn tokenizer_free(handle: TokenizerHandle) {
    take_tokenizer(handle);
}

/// Add a token to the vocabulary
#[no_mangle]
pub extern "C" fn tokenizer_add_token(
    handle: TokenizerHandle,
    token: *const u8,
    token_len: usize,
    id: u32,
) {
    if token.is_null() || token_len == 0 {
        return;
    }

    let token_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(token, token_len)) {
            Ok(s) => s,
            Err(_) => return,
        }
    };

    init_store();
    let mut store = TOKENIZER_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        if let Some(tokenizer) = map.get_mut(&handle) {
            tokenizer.add_token(token_str, id);
        }
    }
}

/// Add a merge rule
#[no_mangle]
pub extern "C" fn tokenizer_add_merge(
    handle: TokenizerHandle,
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) {
    if left.is_null() || right.is_null() || left_len == 0 || right_len == 0 {
        return;
    }

    let left_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(left, left_len)) {
            Ok(s) => s.to_string(),
            Err(_) => return,
        }
    };

    let right_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(right, right_len)) {
            Ok(s) => s.to_string(),
            Err(_) => return,
        }
    };

    init_store();
    let mut store = TOKENIZER_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        if let Some(tokenizer) = map.get_mut(&handle) {
            tokenizer.merges.push((left_str, right_str));
        }
    }
}

/// Set special tokens
#[no_mangle]
pub extern "C" fn tokenizer_set_special_tokens(
    handle: TokenizerHandle,
    pad_token: i64,
    eos_token: i64,
    bos_token: i64,
    unk_token: i64,
) {
    init_store();
    let mut store = TOKENIZER_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        if let Some(tokenizer) = map.get_mut(&handle) {
            tokenizer.pad_token = if pad_token >= 0 {
                Some(pad_token as u32)
            } else {
                None
            };
            tokenizer.eos_token = if eos_token >= 0 {
                Some(eos_token as u32)
            } else {
                None
            };
            tokenizer.bos_token = if bos_token >= 0 {
                Some(bos_token as u32)
            } else {
                None
            };
            tokenizer.unk_token = if unk_token >= 0 {
                Some(unk_token as u32)
            } else {
                None
            };
        }
    }
}

// ============================================================================
// Tokenizer Info
// ============================================================================

/// Get vocabulary size
#[no_mangle]
pub extern "C" fn tokenizer_vocab_size(handle: TokenizerHandle) -> usize {
    get_tokenizer(handle, |t| t.map(|t| t.vocab_size()).unwrap_or(0))
}

/// Get token ID for a token string
#[no_mangle]
pub extern "C" fn tokenizer_token_to_id(
    handle: TokenizerHandle,
    token: *const u8,
    token_len: usize,
) -> i64 {
    if token.is_null() || token_len == 0 {
        return -1;
    }

    let token_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(token, token_len)) {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    get_tokenizer(handle, |t| {
        t.and_then(|t| t.token_to_id(token_str))
            .map(|id| id as i64)
            .unwrap_or(-1)
    })
}

/// Get PAD token ID (-1 if not set)
#[no_mangle]
pub extern "C" fn tokenizer_pad_token_id(handle: TokenizerHandle) -> i64 {
    get_tokenizer(handle, |t| {
        t.and_then(|t| t.pad_token)
            .map(|id| id as i64)
            .unwrap_or(-1)
    })
}

/// Get EOS token ID (-1 if not set)
#[no_mangle]
pub extern "C" fn tokenizer_eos_token_id(handle: TokenizerHandle) -> i64 {
    get_tokenizer(handle, |t| {
        t.and_then(|t| t.eos_token)
            .map(|id| id as i64)
            .unwrap_or(-1)
    })
}

/// Get BOS token ID (-1 if not set)
#[no_mangle]
pub extern "C" fn tokenizer_bos_token_id(handle: TokenizerHandle) -> i64 {
    get_tokenizer(handle, |t| {
        t.and_then(|t| t.bos_token)
            .map(|id| id as i64)
            .unwrap_or(-1)
    })
}

// ============================================================================
// Encoding/Decoding
// ============================================================================

/// Encode text to token IDs
/// Returns number of tokens written to output buffer
#[no_mangle]
pub extern "C" fn tokenizer_encode(
    handle: TokenizerHandle,
    text: *const u8,
    text_len: usize,
    output: *mut u32,
    output_capacity: usize,
) -> usize {
    if text.is_null() || output.is_null() || text_len == 0 || output_capacity == 0 {
        return 0;
    }

    let text_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(text, text_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    get_tokenizer(handle, |t| {
        if let Some(tokenizer) = t {
            let tokens = tokenizer.encode(text_str);
            let n = tokens.len().min(output_capacity);
            unsafe {
                for (i, &token) in tokens.iter().take(n).enumerate() {
                    *output.add(i) = token;
                }
            }
            n
        } else {
            0
        }
    })
}

/// Decode token IDs to text
/// Returns length of decoded text written to output buffer
#[no_mangle]
pub extern "C" fn tokenizer_decode(
    handle: TokenizerHandle,
    input: *const u32,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
) -> usize {
    if input.is_null() || output.is_null() || input_len == 0 || output_capacity == 0 {
        return 0;
    }

    let ids: Vec<u32> = unsafe { std::slice::from_raw_parts(input, input_len).to_vec() };

    get_tokenizer(handle, |t| {
        if let Some(tokenizer) = t {
            let text = tokenizer.decode(&ids);
            let bytes = text.as_bytes();
            let n = bytes.len().min(output_capacity);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, n);
            }
            n
        } else {
            0
        }
    })
}

// ============================================================================
// Text Preprocessing
// ============================================================================

/// Normalize text to NFC form
#[no_mangle]
pub extern "C" fn text_normalize_nfc(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
) -> usize {
    if input.is_null() || output.is_null() || input_len == 0 || output_capacity == 0 {
        return 0;
    }

    let input_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(input, input_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    let normalized: String = input_str.nfc().collect();
    let bytes = normalized.as_bytes();
    let n = bytes.len().min(output_capacity);

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, n);
    }

    n
}

/// Convert text to lowercase
#[no_mangle]
pub extern "C" fn text_lowercase(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
) -> usize {
    if input.is_null() || output.is_null() || input_len == 0 || output_capacity == 0 {
        return 0;
    }

    let input_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(input, input_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    let lowered = input_str.to_lowercase();
    let bytes = lowered.as_bytes();
    let n = bytes.len().min(output_capacity);

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, n);
    }

    n
}

/// Clean whitespace (collapse multiple spaces, trim)
#[no_mangle]
pub extern "C" fn text_clean_whitespace(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
) -> usize {
    if input.is_null() || output.is_null() || input_len == 0 || output_capacity == 0 {
        return 0;
    }

    let input_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(input, input_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    let cleaned: String = input_str.split_whitespace().collect::<Vec<_>>().join(" ");

    let bytes = cleaned.as_bytes();
    let n = bytes.len().min(output_capacity);

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, n);
    }

    n
}

// ============================================================================
// Text Chunking (for RAG)
// ============================================================================

/// Split text into chunks with overlap
/// Returns number of chunks, fills chunk_starts and chunk_lens arrays
#[no_mangle]
pub extern "C" fn text_split_chunks(
    input: *const u8,
    input_len: usize,
    chunk_size: usize,
    overlap: usize,
    chunk_starts: *mut usize,
    chunk_lens: *mut usize,
    max_chunks: usize,
) -> usize {
    if input.is_null() || chunk_starts.is_null() || chunk_lens.is_null() {
        return 0;
    }
    if chunk_size == 0 || chunk_size <= overlap || max_chunks == 0 {
        return 0;
    }

    let input_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(input, input_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    // Split by sentences first for cleaner chunks
    let sentences: Vec<&str> = input_str
        .split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();

    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut current_start = 0;
    let mut current_len = 0;
    let mut last_end = 0;

    for sentence in sentences {
        let sentence_trimmed = sentence.trim();
        if sentence_trimmed.is_empty() {
            continue;
        }

        // Find this sentence in the original text
        if let Some(pos) = input_str[last_end..].find(sentence_trimmed) {
            let abs_pos = last_end + pos;
            let sentence_len = sentence_trimmed.len();

            if current_len == 0 {
                current_start = abs_pos;
                current_len = sentence_len;
            } else if current_len + sentence_len + 1 <= chunk_size {
                current_len = abs_pos + sentence_len - current_start;
            } else {
                // Save current chunk
                if chunks.len() < max_chunks {
                    chunks.push((current_start, current_len));
                }

                // Start new chunk with overlap
                if overlap > 0 && current_len > overlap {
                    current_start = current_start + current_len - overlap;
                    current_len = overlap + sentence_len + 1;
                } else {
                    current_start = abs_pos;
                    current_len = sentence_len;
                }
            }

            last_end = abs_pos + sentence_len;
        }
    }

    // Don't forget the last chunk
    if current_len > 0 && chunks.len() < max_chunks {
        chunks.push((current_start, current_len));
    }

    // Fill output arrays
    for (i, (start, len)) in chunks.iter().enumerate() {
        unsafe {
            *chunk_starts.add(i) = *start;
            *chunk_lens.add(i) = *len;
        }
    }

    chunks.len()
}

/// Count words in text
#[no_mangle]
pub extern "C" fn text_word_count(input: *const u8, input_len: usize) -> usize {
    if input.is_null() || input_len == 0 {
        return 0;
    }

    let input_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(input, input_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    input_str.split_whitespace().count()
}

/// Count characters in text (Unicode-aware)
#[no_mangle]
pub extern "C" fn text_char_count(input: *const u8, input_len: usize) -> usize {
    if input.is_null() || input_len == 0 {
        return 0;
    }

    let input_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(input, input_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    input_str.chars().count()
}

// ============================================================================
// Plugin Registration
// ============================================================================

zrtl_plugin! {
    name: "text",
    symbols: [
        // Tokenizer creation
        ("$Text$tokenizer_create", tokenizer_create),
        ("$Text$tokenizer_free", tokenizer_free),
        ("$Text$tokenizer_add_token", tokenizer_add_token),
        ("$Text$tokenizer_add_merge", tokenizer_add_merge),
        ("$Text$tokenizer_set_special_tokens", tokenizer_set_special_tokens),

        // Tokenizer info
        ("$Text$tokenizer_vocab_size", tokenizer_vocab_size),
        ("$Text$tokenizer_token_to_id", tokenizer_token_to_id),
        ("$Text$tokenizer_pad_token_id", tokenizer_pad_token_id),
        ("$Text$tokenizer_eos_token_id", tokenizer_eos_token_id),
        ("$Text$tokenizer_bos_token_id", tokenizer_bos_token_id),

        // Encoding/decoding
        ("$Text$tokenizer_encode", tokenizer_encode),
        ("$Text$tokenizer_decode", tokenizer_decode),

        // Text preprocessing
        ("$Text$normalize_nfc", text_normalize_nfc),
        ("$Text$lowercase", text_lowercase),
        ("$Text$clean_whitespace", text_clean_whitespace),

        // Text utilities
        ("$Text$split_chunks", text_split_chunks),
        ("$Text$word_count", text_word_count),
        ("$Text$char_count", text_char_count),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_lifecycle() {
        let handle = tokenizer_create();
        assert_ne!(handle, TOKENIZER_NULL);

        // Add tokens
        tokenizer_add_token(handle, b"hello".as_ptr(), 5, 1);
        tokenizer_add_token(handle, b"world".as_ptr(), 5, 2);

        assert_eq!(tokenizer_vocab_size(handle), 2);
        assert_eq!(tokenizer_token_to_id(handle, b"hello".as_ptr(), 5), 1);
        assert_eq!(tokenizer_token_to_id(handle, b"world".as_ptr(), 5), 2);
        assert_eq!(tokenizer_token_to_id(handle, b"unknown".as_ptr(), 7), -1);

        tokenizer_free(handle);
    }

    #[test]
    fn test_encode_decode() {
        let handle = tokenizer_create();

        // Create simple character-level tokenizer
        tokenizer_add_token(handle, b"h".as_ptr(), 1, 0);
        tokenizer_add_token(handle, b"e".as_ptr(), 1, 1);
        tokenizer_add_token(handle, b"l".as_ptr(), 1, 2);
        tokenizer_add_token(handle, b"o".as_ptr(), 1, 3);

        // Encode "hello"
        let mut output = [0u32; 10];
        let n = tokenizer_encode(handle, b"hello".as_ptr(), 5, output.as_mut_ptr(), 10);

        assert_eq!(n, 5);
        assert_eq!(output[0], 0); // h
        assert_eq!(output[1], 1); // e
        assert_eq!(output[2], 2); // l
        assert_eq!(output[3], 2); // l
        assert_eq!(output[4], 3); // o

        // Decode back
        let mut text = [0u8; 10];
        let text_len = tokenizer_decode(handle, output.as_ptr(), 5, text.as_mut_ptr(), 10);

        assert_eq!(text_len, 5);
        assert_eq!(&text[..5], b"hello");

        tokenizer_free(handle);
    }

    #[test]
    fn test_bpe_merge() {
        let handle = tokenizer_create();

        // Character tokens
        tokenizer_add_token(handle, b"h".as_ptr(), 1, 0);
        tokenizer_add_token(handle, b"i".as_ptr(), 1, 1);
        tokenizer_add_token(handle, b"hi".as_ptr(), 2, 2);

        // Add merge rule: h + i -> hi
        tokenizer_add_merge(handle, b"h".as_ptr(), 1, b"i".as_ptr(), 1);

        // Encode "hi" - should use merged token
        let mut output = [0u32; 10];
        let n = tokenizer_encode(handle, b"hi".as_ptr(), 2, output.as_mut_ptr(), 10);

        assert_eq!(n, 1);
        assert_eq!(output[0], 2); // Should be the merged "hi" token

        tokenizer_free(handle);
    }

    #[test]
    fn test_text_lowercase() {
        let input = b"Hello WORLD";
        let mut output = [0u8; 20];

        let n = text_lowercase(input.as_ptr(), 11, output.as_mut_ptr(), 20);

        assert_eq!(n, 11);
        assert_eq!(&output[..n], b"hello world");
    }

    #[test]
    fn test_text_clean_whitespace() {
        let input = b"  hello   world  ";
        let mut output = [0u8; 20];

        let n = text_clean_whitespace(input.as_ptr(), 17, output.as_mut_ptr(), 20);

        assert_eq!(&output[..n], b"hello world");
    }

    #[test]
    fn test_word_count() {
        let text = b"hello world how are you";
        assert_eq!(text_word_count(text.as_ptr(), 23), 5);
    }

    #[test]
    fn test_char_count() {
        // Test with ASCII
        assert_eq!(text_char_count(b"hello".as_ptr(), 5), 5);

        // Test with UTF-8 (emoji: 👋 is 4 bytes but 1 character)
        let emoji = "👋";
        assert_eq!(text_char_count(emoji.as_ptr(), emoji.len()), 1);
    }
}
