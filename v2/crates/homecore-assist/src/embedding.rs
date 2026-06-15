//! Deterministic text embedding for semantic intent matching.
//!
//! No ML model dependency: utterances are embedded with the classic
//! **feature-hashing** (hashing-vectorizer) technique. Each n-gram feature is
//! hashed into a fixed-width vector; a second sign-hash decides whether the
//! feature adds or subtracts, which keeps the expected dot-product unbiased
//! under collisions. The vector is L2-normalised so that cosine similarity is
//! a clean `1 - distance`.
//!
//! Features used per utterance:
//! - **word unigrams** — whole tokens after lowercasing/trimming punctuation.
//! - **character trigrams** — sliding 3-grams over each padded token, which
//!   gives partial-overlap credit ("kitchen" ~ "kitchens") and robustness to
//!   small lexical variation.
//!
//! This is intentionally *lexical-semantic*: paraphrases that share tokens
//! ("turn on the light" vs "turn on the kitchen light") land close together,
//! while unrelated utterances ("play jazz music") land far apart. It is a real,
//! reproducible similarity signal — not a hash that ignores meaning.
//!
//! The output dimension matches [`EMBEDDING_DIM`] and is consumed directly by
//! the exact in-memory cosine k-NN in `crate::semantic_recognizer`.

/// Dimensionality of the hashed embedding space.
///
/// 256 buckets keeps collisions low for the small intent vocabularies HOMECORE
/// deals with while staying cheap to index in HNSW.
pub const EMBEDDING_DIM: usize = 256;

// FNV-1a 64 constants — small, fast, well-distributed for feature hashing.
const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a64(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = seed;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

/// Accumulate one hashed feature into `acc` with signed weight.
#[inline]
fn add_feature(acc: &mut [f32], feature: &[u8], weight: f32) {
    let h = fnv1a64(FNV_OFFSET_BASIS_64, feature);
    let bucket = (h % EMBEDDING_DIM as u64) as usize;
    // Independent sign hash (different seed) → unbiased under collisions.
    let sign = if fnv1a64(0x100, feature) & 1 == 0 { 1.0 } else { -1.0 };
    acc[bucket] += sign * weight;
}

/// Normalise text: lowercase, keep alphanumerics, split on everything else.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .collect()
}

/// Embed an utterance into a deterministic, L2-normalised vector.
///
/// Returns a zero vector only for input with no alphanumeric content.
pub fn embed(text: &str) -> Vec<f32> {
    let mut acc = vec![0.0_f32; EMBEDDING_DIM];
    let tokens = tokenize(text);

    for tok in &tokens {
        // Word unigram — weighted higher than sub-word features.
        add_feature(&mut acc, format!("w:{tok}").as_bytes(), 1.5);

        // Character trigrams over a padded token so prefixes/suffixes count.
        let padded: Vec<char> = format!("^{tok}$").chars().collect();
        if padded.len() >= 3 {
            for window in padded.windows(3) {
                let gram: String = window.iter().collect();
                add_feature(&mut acc, format!("c:{gram}").as_bytes(), 1.0);
            }
        }
    }

    l2_normalise(&mut acc);
    acc
}

/// L2-normalise in place; no-op for the zero vector.
fn l2_normalise(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two equal-length vectors (dot product of unit vectors).
///
/// Exposed for tests and for callers that want similarity without round-tripping
/// through the HNSW index.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_has_correct_dim() {
        assert_eq!(embed("turn on the light").len(), EMBEDDING_DIM);
    }

    #[test]
    fn embedding_is_deterministic() {
        assert_eq!(embed("turn on the light"), embed("turn on the light"));
    }

    #[test]
    fn embedding_is_unit_norm() {
        let v = embed("turn on the kitchen light");
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-4, "norm^2 = {norm_sq}");
    }

    #[test]
    fn empty_input_is_zero_vector() {
        let v = embed("!!! ???");
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn paraphrase_is_more_similar_than_unrelated() {
        let exemplar = embed("turn on the light");
        let paraphrase = embed("turn on the kitchen light");
        let unrelated = embed("play some jazz music");

        let sim_para = cosine_similarity(&exemplar, &paraphrase);
        let sim_unrel = cosine_similarity(&exemplar, &unrelated);

        assert!(
            sim_para > sim_unrel,
            "paraphrase ({sim_para:.3}) must beat unrelated ({sim_unrel:.3})"
        );
        // Real, non-trivial separation.
        assert!(sim_para > 0.5, "paraphrase similarity too low: {sim_para:.3}");
        assert!(sim_unrel < 0.3, "unrelated similarity too high: {sim_unrel:.3}");
    }

    #[test]
    fn embeddings_are_structurally_finite() {
        // SECURITY (NaN-poisoning): the embedding path takes only `&str` and
        // produces values via FNV feature-hashing + a guarded L2 normalise.
        // There is NO external float input and NO unguarded division, so a
        // crafted utterance cannot inject NaN/±Inf into a vector and poison the
        // cosine k-NN match. Prove every component is finite across adversarial
        // inputs (empty, punctuation-only, unicode, very long, control chars).
        for s in [
            "",
            "!!! ???",
            "turn on the kitchen light",
            "🔥🔥🔥 \u{0}\u{1}\u{7f} mix",
            &"x".repeat(10_000),
            "NaN inf -inf 1e999",
        ] {
            let v = embed(s);
            assert_eq!(v.len(), EMBEDDING_DIM);
            assert!(
                v.iter().all(|x| x.is_finite()),
                "embedding of {s:?} contained a non-finite component"
            );
        }
    }

    #[test]
    fn cosine_with_zero_vector_is_finite_not_nan() {
        // SECURITY (NaN-poisoning): an empty/punctuation-only utterance embeds
        // to the zero vector. Cosine against any exemplar must be a finite 0.0,
        // never NaN — so a below-threshold comparison stays well-defined and the
        // recognizer falls through (no action) rather than matching on garbage.
        let zero = embed("!!! ???");
        let real = embed("turn on the light");
        let sim = cosine_similarity(&zero, &real);
        assert!(sim.is_finite(), "cosine vs zero vector must be finite, got {sim}");
        assert_eq!(sim, 0.0, "dot product with the zero vector is exactly 0");
    }

    #[test]
    fn identical_text_is_similarity_one() {
        let a = embed("lock the front door");
        let b = embed("lock the front door");
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-4, "sim = {sim}");
    }
}
