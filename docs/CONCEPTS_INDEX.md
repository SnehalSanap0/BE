# Concept guide — index

This is a living set of plain-language explanations for the ideas used across this project — written to build actual understanding you can reason from and defend under cross-questions, not dictionary definitions. Each one lives next to the code it explains, so it stays easy to find while you're reading that part of the project:

| Folder | What it explains |
|---|---|
| [`preprocessing/CONCEPTS.md`](../preprocessing/CONCEPTS.md) | Mono conversion, resampling, silence trimming, normalization, fixed-length padding — and why padding created this project's biggest confound |
| [`feature_extraction/CONCEPTS.md`](../feature_extraction/CONCEPTS.md) | Spectrograms, mel-spectrograms, MFCC, LFCC — why audio gets turned into an "image" at all |
| [`dataset/CONCEPTS.md`](../dataset/CONCEPTS.md) | Train/val/test splits, stratification, and why the standard random split can hide a shortcut here |
| [`models/CONCEPTS.md`](../models/CONCEPTS.md) | Transfer learning, pretrained backbones, pooling, embeddings, the optional BiLSTM head |
| [`evaluation/CONCEPTS.md`](../evaluation/CONCEPTS.md) | Accuracy, precision, recall, F1, ROC-AUC, EER, confusion matrix, decision threshold |
| [`explainability/CONCEPTS.md`](../explainability/CONCEPTS.md) | Grad-CAM, heatmaps — what an "explanation" is and isn't |
| [`scripts/CONCEPTS.md`](../scripts/CONCEPTS.md) | Shortcut learning, confounds, cross-source holdout testing, domain shift, calibration, embedding-space/t-SNE/silhouette analysis |
| [`deployment/CONCEPTS.md`](../deployment/CONCEPTS.md) | Training vs. inference, why the model loads once, Flask API vs. Gradio UI |

## How to use this for your defense / viva

Each file ends with a **"Rational summary"** section. Read the main text first for the intuition, then use those Q&A sections to rehearse explaining it out loud in your own words.

The throughline across almost every file is the same real finding from this project, seen from a different angle each time:

1. **`preprocessing/`** — padding to a fixed duration introduced a source-correlated confound (unequal amounts of added silence).
2. **`dataset/`** — the standard random split let that confound hide, because every source appeared in both train and test.
3. **`scripts/`** — cross-source holdout testing exposed it (performance varied 3.7%–40.5% EER depending on which sources were held out), and embedding/silhouette analysis showed *why*, by measuring how strongly the model's internal representation encodes source identity vs. the intended label.
4. **`evaluation/`** — this is exactly why the project reports EER/AUC/precision/recall/confusion matrices side by side instead of a single accuracy number: each metric exposes a different part of this story (a pooled-split model scored perfectly on accuracy while still carrying the shortcut).
5. **`explainability/`** — Grad-CAM offers a second, independent way to check the same question by inspecting *what the model looked at* on individual clips.
6. **`deployment/`** — and it's exactly this generalization gap that explains why an unfamiliar real-world input (like an ElevenLabs Hindi TTS clip, from a synthesis system never seen in training) can be misclassified with high confidence, even though the model performs very well on its own benchmark.

If you can explain that six-step chain in your own words, using the vocabulary from each file, you can defend this project's methodology under most cross-questioning.

## Keeping this updated

When a new term, metric, or technique gets introduced into the project, add it to the most relevant folder's `CONCEPTS.md` (create one if that folder doesn't have one yet) in the same style: plain intuition first, a concrete tie-in to this project's actual numbers/findings, then a short cross-question Q&A at the end — and add a row to the table above.
