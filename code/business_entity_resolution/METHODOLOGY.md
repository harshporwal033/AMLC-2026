# Methodology – Business Entity Resolution

Draft for `Documentation_template.md` (copy each section into the matching heading of the
official template). Numbers marked *dev* are on our held-out validation set; *LB* are public
leaderboard scores.

## 1. Problem framing

Source 1 (S1) is deduplicated; every S2/S3 record matches **at most one** S1 entity (verified
on the training labels), and ~26% of train S2/S3 records match none. We therefore solve the task
from the S2/S3 side: for every S2/S3 record find its best S1 candidate, decide whether to accept
it, and group accepted records by S1. This one-to-one structure removes a whole class of false
merges.

The metric is macro F0.5 per S1 entity, with singletons counting (empty prediction = 1.0).
Two facts shaped the design:

- **Decoys are hard negatives.** Unmatched S2/S3 records are near-copies of real S1 businesses
  with one detail changed (house number ±1…10, legal form LLP ↔ Private Limited, one name word
  swapped), while true copies carry *different* noise (typos, OCR 0/O and l/i, dropped leading
  digits, leading zeros, abbreviations, transliteration into Indic scripts, junk names, missing
  addresses).
- **Test has ~1.9× more decoys per S1 than train** (5.75 vs 4.7 records per S1 with the same
  number of true matches per S1). All models and the decision rule are tuned for that density
  (decoy rows weighted ×1.9 in training and in validation scoring).

## 2. Validation design

A random 2% of train S1 entities (44k) form the dev set. It contains every S2/S3 record whose
true match is a dev entity **or** that has a dev entity in its top-2 candidates, with all of that
record's candidates. So false matches landing on a dev entity from anywhere in the data are
counted, as on test (an earlier dev set built by sampling records hid ~20× of these and was
over-optimistic by 0.04). All training uses grouped out-of-fold predictions (groups = true S1),
and the macro F0.5 is computed exactly as the leaderboard does, with decoy false positives
weighted ×1.9. Cross-encoder training pairs come from **other** train entities, so its dev
scores are out-of-sample.

## 3. Normalisation

`anyascii` transliteration (Indic scripts → Latin), lower-casing, punctuation removal, `&` →
`and`, domain suffixes and `(ID: …)` / `#123` junk removed, "trading as / DBA" handled, leading
honorifics removed, legal-form words separated from the core name, address abbreviations unified
(street/st/saint, road/rd, …), `null` tokens dropped, house-number letter suffixes stripped.
Country is used only to partition the search; it is never a model feature, so France (absent from
train) is handled like any other country.

## 4. Candidate generation (blocking)

Per country, for every S2/S3 record, the top-10 S1 records from each of two views:

1. **Word view** – sparse TF-IDF over rare words of name + address (words in >2% of a country's
   S1 records dropped), exact top-k by sparse matrix product (`sparse_dot_topn`).
2. **Embedding view** – `multilingual-e5-small` sentence embeddings of the *raw* name + address
   (original script, so Indic names match their English form), exact cosine top-k on GPU.

Recall of the true S1 within the candidate set: **98.6%** on train (≈18 candidates per record);
most remaining misses have an empty address and a name shared by several S1 entities.
(Character-trigram views were tried: +1% recall at ~10× the cost; replaced by the embedding view.)

## 5. Stage 1 – pair model (LightGBM)

46 features per (record, candidate) pair:
- view scores / ranks, gap to the record's best candidate, margin best–second;
- string similarities (RapidFuzz token-set / token-sort / partial ratio, Jaro-Winkler) on
  normalised names and addresses;
- **decoy-sensitive features**: legal-form agreement (both present / equal / different),
  house-number relation (equal, truncated, close |Δ|≤10, far; leading zeros removed),
  name words that differ as typos vs. genuinely different words (Jaro-Winkler of unmatched
  tokens), number-set overlap;
- junk-name signals (share of name tokens seen in S1, rarest-token frequency), lengths,
  non-Latin flag, source, ambiguity counts (candidates of the record with near-identical name /
  address).
Binary LightGBM, grouped 3-fold OOF, decoy rows weighted ×1.9. Output p1 per pair; each record
keeps its best candidate (top-1) and runner-up.

## 6. Cross-encoder

`Qwen3-Reranker-0.6B` fine-tuned as a pair classifier on 1M (record, S1) text pairs
(`name | address`, raw script) from non-dev train entities: all true pairs + hard negatives
(top-4 of either blocking view), BCE, bf16, 1 epoch, lr 2e-5 (validation AUC 0.9996).
It scores the record's top-1 and runner-up S1 **only for uncertain records** (stage-1 p1 in
(0.02, 0.995); ~33% of records), on both dev and test with the same rule. Earlier iteration:
a cross-fitted `multilingual-e5-small` cross-encoder (MIT) on all records.

## 7. Stage 2 – competition model (LightGBM)

One row per S2/S3 record (its top-1 S1), features = stage-1 features + p1 +
- **competition among records claiming the same S1** (rank of this record, gap to the strongest
  other claimant, sum / count of other claimants' p1);
- **runner-up** (p of the second-best S1, whether that S1 is already claimed by a stronger
  record);
- **coherence** (name / address / number agreement with the other claimants of the same S1);
- **cross-encoder** score of top-1 and runner-up and their gap.
Trained on dev entities (complete claimant sets), grouped OOF by S1, decoys ×1.9.

## 8. Decision rule

Per S1, the accepted records are the top-m by stage-2 probability, where m (0…n) maximises the
plug-in expected F0.5 of that S1 (m = 0 scores P(no true match) = ∏(1−p)); probabilities are
sharpened by a power tuned on dev. This treats a lone uncertain match on an otherwise empty S1
(a singleton risk) differently from the same probability next to confident matches.

## 9. Results

| Version | dev (test-like F0.5) | LB |
|---|---|---|
| stage 1 only, early features | 0.9465 | 0.932 |
| + decoy features, stage 2 (competition) | 0.9685 | 0.950 |
| + coherence + e5 cross-encoder (25% dev sample) | 0.9832 | 0.973 |
| full dev set, no cross-encoder | 0.9717 | – |
| + Qwen3 cross-encoder | *fill in* | *fill in* |

Key contributions (dev): decoy-sensitive features +0.019; stage-2 competition +0.008;
cross-encoder +0.011 (e5-small) / *fill in* (Qwen3).

## 10. Compliance

No external data, APIs, geocoding or lookups; only the provided files. Pretrained weights:
`intfloat/multilingual-e5-small` (MIT, 118M params) and `Qwen/Qwen3-Reranker-0.6B`
(Apache-2.0, 0.6B params), both within the MIT/Apache-2.0 and ≤8B rules.
