# Understanding audio preprocessing

## Why preprocessing has to happen at all

The raw audio files pulled from different source datasets are not directly comparable: they can arrive at different sample rates, in stereo or mono, with wildly different amounts of leading/trailing silence, and at different loudness levels. Feeding this inconsistency straight into a model would let *those* differences become part of what the model learns to key on, instead of the genuine real-vs-fake distinction. Preprocessing exists to strip out every difference that isn't supposed to matter, so whatever differences remain are more likely to be the ones that actually matter — though, as this project's own research log shows, preprocessing can only reduce this risk, not eliminate it entirely (see the silence-padding confound below).

## Mono conversion — one consistent number of audio channels

Some source files are stereo (2 channels); this project always wants exactly 1. Converting to mono (by averaging the channels) removes a difference that's purely about how a file happened to be recorded, not about its content.

## Resampling — one consistent sample rate

"Sample rate" is how many amplitude measurements are taken per second (e.g. 16,000 per second = 16kHz). Different datasets are often recorded/exported at different sample rates. If left unmatched, a model could partially learn "this sample rate pattern = this source = this label" — a shortcut, not genuine understanding. Resampling everything to a single fixed rate (16kHz here) removes that possibility, and also matches what the feature-extraction step (frequency-axis calculations) expects as input.

## Silence trimming — removing dead air at the start/end

Recordings often have a bit of silence before speech starts and after it ends, and the *amount* of that silence has nothing to do with whether the speech itself is real or synthetic — it's an artifact of how each dataset was recorded or exported. Trimming it (via `librosa.effects.trim`, thresholded by `top_db`) removes leading/trailing near-silence, keeping the analysis focused on the part of the signal that actually contains speech.

## Peak normalization — one consistent loudness reference

Different recordings can simply be louder or quieter than each other for reasons unrelated to real-vs-fake (recording equipment, source volume, export settings). Peak normalization rescales every clip so its loudest point hits the same reference level (amplitude 1.0), removing raw-loudness as a possible shortcut signal.

## Pad/truncate to a fixed duration — why this step is the most consequential one for this project

Models generally need a fixed-size input. Since real speech clips and TTS clips naturally have almost any duration, every clip is forced to exactly 4 seconds — clips longer than that get cut off, clips shorter than that get **zero-padded** at the end (silence added).

**This is exactly the step that created the biggest confound found in this project.** Trimming silence (above) only removes *actual* recorded silence; it does nothing to prevent *this* step from adding new, artificial silence back on afterward, to reach the fixed 4-second length. A source dataset whose natural clips run much shorter than 4 seconds (e.g. Common Voice) ends up with far more of this added padding than a source whose clips are naturally closer to 4 seconds already (e.g. IndicSynth) — a difference that has nothing to do with real vs. fake, purely a side effect of each source's typical recording length interacting with this one fixed-duration design choice. `scripts/confound_check.py` measures exactly this ("trailing-zero ratio"), and it's the single largest confound this project has found.

## Why the *same* code path is used for training and for a live prediction

`preprocessing.transforms.preprocess_audio` is called identically by the offline batch pipeline (`preprocessing/pipeline.py`) and by the online single-clip inference path (`deployment/app.py`, `deployment/gradio_app.py`, via `explainability/explain.py`). If these two paths ever preprocessed audio even slightly differently, a live prediction could behave unpredictably relative to what the model was actually trained and evaluated on — a subtle but common source of "works in testing, fails in production" bugs. Sharing one function guarantees that can't happen here.

## Rational summary

- **"Why does padding matter more than any other preprocessing step?"** — because it's the one step whose *amount* systematically differs by source dataset (due to each source's differing natural clip length), giving the model a non-genuine cue to potentially exploit, whereas the other steps (mono/resample/normalize) remove differences rather than introducing a new, source-correlated one.
- **"Could this have been avoided?"** — yes, in principle (e.g. by cropping/looping instead of zero-padding, or by choosing sources with more similar natural durations) — this is exactly the kind of fix a controlled follow-up experiment (silence-normalize everything, then re-run the cross-source factorial) would test, as flagged as future work in the research log.
- **"Why fix the duration at all instead of letting clips vary?"** — the model architecture (a CNN expecting a fixed-size 224×224 input after feature extraction) requires it; a variable-length design is possible in principle (e.g. via more elaborate padding/masking or a fully recurrent model) but adds complexity this project's baseline architecture doesn't use.
