# Understanding the diagnostic scripts

This folder answers one question from several angles: **is the model actually detecting synthetic speech, or has it found an easier shortcut that happens to work on this dataset?** Each concept below is a different lens on that same question.

## Shortcut learning (the central idea everything else supports)

A neural network doesn't know what "real vs. fake" is supposed to mean — it just finds *any* pattern in the training data that lets it minimize error. If there's an easier pattern available than the one you intended, it will find and use that instead, with no warning to you. This is called a **shortcut** (also "spurious correlation" in the literature).

Concrete example from this project: real clips and fake clips didn't just differ in "genuine vs. synthetic speech" — they also happened to differ in things like how much silence padding each recording had, purely because of how each source dataset was recorded. A model can hit high accuracy by learning "clips with lots of trailing silence are usually real" without ever learning anything about synthesis artifacts. It still "works" on data shaped like its training set, and quietly fails the moment that superficial pattern doesn't hold (e.g., a differently-padded fake clip).

This isn't a bug in the code — it's a fundamental risk in supervised learning whenever a shortcut and the true signal are both present and correlated with the label. The only way to catch it is to test in a way the shortcut can't survive — which is exactly what cross-source holdout testing does.

## A confound (the technical word for "the thing that made the shortcut possible")

A confound is a variable that differs systematically between your classes for a reason that has nothing to do with the concept you're trying to teach the model. Here, "which source dataset a clip came from" is a confound for "real vs. fake," because each label was built by pooling multiple source datasets that also differ in recording channel, silence patterns, mic quality, etc. — differences the model can exploit as a shortcut.

`scripts/confound_check.py` doesn't test the model at all — it just measures whether these confounding differences actually exist and how large they are, directly from the audio, independent of any model. That's why its findings (e.g., SEA-Spoof carrying ~5.4x more trailing silence than IndicSynth) are trustworthy evidence even before you look at what the model learned: they establish that a shortcut was *available*, which is the necessary precondition for the model being able to exploit it.

## Cross-source (cross-corpus) holdout testing

The standard train/test split in this project's main pipeline splits *randomly across all pooled clips* — so every source dataset appears in both training and testing. That's the setup an ordinary ML course teaches you to use, but it's exactly the setup that lets a shortcut hide: if the model learned "Common Voice clips are real," and Common Voice clips are in both train and test, that shortcut *still works* on the test set and inflates every metric.

Cross-source holdout testing instead trains on some *entire source datasets* and tests only on *completely different, unseen* source datasets. If the model's performance holds up, it must be relying on something more general than "which corpus is this" (since the test corpora were never seen). If performance collapses, the earlier "good" numbers were partly or entirely a shortcut. This is the same logic as testing a student on questions from a textbook chapter they've never opened, instead of the one they studied from — if they still do well, they've actually learned the underlying subject, not memorized the specific examples.

## Domain shift

A general term for "the data you're testing on doesn't look like the data you trained on" — different recording conditions, different TTS engine, different language, different microphone, etc. Cross-source holdout testing is a *controlled, deliberate* domain shift (we know exactly which sources are held out). A real-world upload (like an ElevenLabs clip never seen in training at all) is an *uncontrolled* domain shift — the model has no guarantee of handling it well, and this project's own research log documents that it often doesn't.

## Calibration vs. discrimination (why a model can be "right" but still "wrong")

These are two different properties a model can have, and it's possible to have one without the other — this distinction explains several surprising results in this project's log.

- **Discrimination** = does the model's internal score genuinely rank fakes above reals? (Measured by ROC-AUC — see `evaluation/CONCEPTS.md`.)
- **Calibration** = does the model's score of "0.7" actually correspond to roughly 70% real-world likelihood, at the threshold you're using to make decisions?

A model can have good discrimination but bad calibration under domain shift: it still "knows," in a ranking sense, which clips look more fake than others, but its absolute scores have drifted (e.g., compressed toward "real") so the usual 0.5 cutoff no longer makes sense for that domain. That's exactly what happened in the `swapped` rotation: ROC-AUC stayed reasonably high (0.876) while accuracy at the default threshold collapsed to near-chance (50.3%) — a calibration failure, not a total inability to detect fakes.

## Embedding space, and why we visualize it

Deep inside the model, before the final classification decision, every clip gets converted into a list of 1280 numbers (a vector) — its **embedding**. Similar-sounding clips tend to land near each other in this 1280-dimensional space; dissimilar ones land far apart. The final classifier is just drawing a boundary through this space.

Humans can't look at 1280 dimensions directly, so **t-SNE** (a dimensionality-reduction technique) squashes each 1280-number embedding down to just 2 numbers (an x, y point) while trying to preserve which points were close together in the original high-dimensional space. Plotting these 2-D points lets you visually check: do points cluster by *label* (what we want) or by *source dataset* (the shortcut)?

## Silhouette score, and the "shortcut_ratio" this project computes

A silhouette score is a number (roughly -1 to 1) that measures how cleanly a set of points separates into distinct groups: high means points in the same group sit close together and are far from points in other groups; low means the groups blur into each other. We compute it twice for the same embeddings — once treating "source dataset" as the grouping, once treating "real/fake label" as the grouping — and divide: `shortcut_ratio = silhouette(source) / silhouette(label)`.

- A **low** ratio means the embedding space separates clips mainly by the intended label, not by source — a good sign.
- A **high** ratio (close to or above 1, as seen in the `fake_swap_only` rotation: 0.956) means the embedding space is separating clips by *which dataset they came from* almost as strongly as by whether they're real or fake — direct, quantitative evidence the model leaned on source identity rather than the intended task, for that particular training pairing.

This is the single most concrete, defensible number in the whole diagnostic suite, because it turns "the model might be using a shortcut" (a suspicion) into "here is a measured number showing how much source-identity structure survives inside the model's own learned representation" (an actual measurement).

## Rational summary

- **"How do you know it's a shortcut and not just genuine but source-correlated learning?"** — no single script proves it alone; the case is built from three independent, converging pieces of evidence: (1) the confound audit shows a shortcut *was available* in the raw signal statistics, (2) cross-source holdout testing shows performance *degrades* when that shortcut can't transfer, and (3) the embedding/silhouette analysis shows the model's *internal representation* actually encodes source identity more strongly exactly when generalization is worst. Agreement across all three, at the same rotation (`fake_swap_only`), is what makes the conclusion credible rather than speculative.
- **"Why t-SNE and not just look at accuracy?"** — accuracy is a single downstream number; t-SNE/silhouette let you look *inside* the model at the representation that produced that number, which is what lets you distinguish "worked because it learned the task" from "worked because it found an easier pattern."
- **"Could this all just be random noise in a small dataset?"** — that's exactly why the diagnostics were built to be consistent across multiple independent measurements (signal statistics AND embedding geometry AND held-out performance) rather than relying on any one of them alone.
