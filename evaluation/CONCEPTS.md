# Understanding the evaluation metrics

This is not a dictionary. It's how to actually *think* about the numbers this project prints, so you can explain them in your own words and survive a cross-question like "why not just use accuracy?"

Everything below assumes: **fake = the thing we're trying to catch** (like a security alarm). This framing makes every metric intuitive instead of abstract.

## The four outcomes (this is the only thing you truly need to memorize)

Every prediction the model makes falls into exactly one of four buckets:

| | Model said "fake" | Model said "real" |
|---|---|---|
| **Actually fake** | ✅ Caught it (True Positive) | ❌ Missed it (False Negative) |
| **Actually real** | ⚠️ False alarm (False Positive) | ✅ Correctly cleared (True Negative) |

The **confusion matrix** you see in outputs is just this table filled in with counts. Nothing more mysterious than that.

Two of these four are the dangerous ones for a security tool:
- **False Negative (missed spoof)** — the worst outcome. A fake clip walks through labeled "real."
- **False Positive (false alarm)** — annoying, wastes a human reviewer's time, but not dangerous.

Notice these two failure modes are *not* equally bad. Any metric that treats them as equally bad (like plain accuracy) is hiding something from you.

## Accuracy — why it's not enough on its own

Accuracy = (everything you got right) / (everything). Simple, but it lies to you in two specific ways relevant to this project:

1. **It doesn't tell you *which* mistakes you're making.** A model that misses 90% of spoofed clips but the dataset is mostly real clips can still show 90%+ accuracy. Accuracy averages over both failure modes as if they were interchangeable — they aren't (see above).
2. **It can be inflated by a shortcut instead of real learning.** Our own `main` pooled-split model hit 100% accuracy — not because it's a perfect spoof detector, but because every source dataset appeared in both training and testing, so the model could partly cheat by recognizing *which recording setup* a clip came from rather than genuinely detecting synthesis artifacts. High accuracy told us nothing about this; other metrics (see cross-source EER below) did.

**Rational takeaway:** accuracy answers "how often was I right," not "was I right for the right reasons" or "which kind of wrong am I."

## Precision — "when I raise the alarm, can I trust it?"

Precision = (true fakes caught) / (everything I flagged as fake). If precision is low, your "fake" alarm cries wolf a lot — most of what it flags turns out to be real. High precision means when the model says "fake," it's usually correct.

## Recall — "of all the fakes that exist, how many did I actually catch?"

Recall = (true fakes caught) / (all fakes that actually existed). This is the metric that matters most for a security use case, because **1 − recall = miss rate** — the fraction of real spoofs that slipped through undetected. In our `original` cross-source rotation, the miss rate was ~4.7% (71 out of 1500 spoof clips slipped through as "real"); in the worst rotation (`swapped`), the miss rate spiked to over 99%.

**Precision and recall trade off against each other** — you can always get recall to 100% by flagging *everything* as fake, but then precision collapses (constant false alarms). Neither number alone tells the full story; that's why we report both.

## F1 — a single number that punishes imbalance between precision and recall

F1 is the harmonic mean of precision and recall. It's high only when *both* are reasonably high — a model that's great at one and terrible at the other gets a mediocre F1, not an inflated one. Useful as a single checkpoint-selection number (this project uses `val_f1` to decide which epoch is "best"), but always look at precision/recall separately too, especially in a security context where the two failure modes aren't equally costly.

## ROC-AUC — "does the model's score genuinely separate fake from real, regardless of where I draw the line?"

Every prediction isn't just "fake" or "real" — underneath, the model outputs a probability score. Precision/recall/F1 depend on where you draw the threshold line (usually 0.5). **ROC-AUC does not** — it asks: if I pick one random real clip and one random fake clip, what's the chance the model's score ranks the fake one higher? A perfect ranker scores 1.0; random guessing scores 0.5.

This matters because a model can have a *terrible* accuracy at the default threshold but still "know" the right answer internally. We saw this directly in the `swapped` rotation: accuracy collapsed to 50.3% (near chance) at the default 0.5 threshold, but ROC-AUC was still 0.8761 — meaning the model's internal ranking was still mostly correct, it was just miscalibrated about where "0.5" should sit for that unfamiliar domain. That's a **calibration problem**, not a **discrimination problem** — and only ROC-AUC (versus accuracy/F1) tells you which one you're dealing with.

## EER (Equal Error Rate) — the standard single number in anti-spoofing research

Borrowed from biometrics/security research (fingerprint/face/voice-spoof detection), not something we invented. EER is the point where the two error types are made equal: the fraction of spoofs missed (false-reject-like) equals the fraction of real clips falsely flagged (false-accept-like), by sliding the decision threshold until those two curves cross. That crossing point's error rate is reported as "the EER."

- Lower is better. 0% would mean a perfect separation exists at some threshold.
- It's threshold-independent in the sense that it *finds* the best possible threshold for balancing the two error types, then reports the error rate there — so it's a fairer single-number summary than accuracy at a fixed, possibly wrong, threshold.
- Reported alongside `eer_threshold` — the actual score cutoff where that balance happens. If `eer_threshold` is oddly close to 0 (as seen in several cross-source rotations), that's the tell-tale sign of the calibration collapse described above: the model's fake-probability output got compressed toward zero for an unfamiliar domain, so even a tiny score counts as "fake" at the balanced point.

## Decision threshold — the one design choice hiding behind every metric except AUC/EER

By default, "probability ≥ 0.5 → predict fake." But 0.5 is just a convention, not a law. Given that a **missed spoof is worse than a false alarm** in this problem, a deployed system might deliberately use a *lower* threshold than 0.5 — catch more fakes at the cost of more false alarms — rather than blindly using the EER-equalizing threshold, which treats both error types as equally costly. This is a judgment call about the *application*, not something a metric can decide for you.

## Rational summary

- **"Why report so many metrics instead of just accuracy?"** — because accuracy conflates two failure modes that aren't equally costly, and can be inflated by a shortcut the model learned instead of genuine detection.
- **"What's the difference between AUC and accuracy/F1?"** — AUC measures whether the model *knows* the right answer (ranking), independent of threshold; accuracy/F1 measure whether it *acts* on that knowledge correctly at one specific threshold. A model can score well on one and poorly on the other — that gap is itself diagnostic (it means "recalibrate," not "retrain from scratch").
- **"Why EER specifically, and not just F1?"** — EER is the field-standard metric in spoof/anti-spoofing literature, which makes our numbers directly comparable to other papers' reported results, and it self-selects the fairest threshold rather than assuming 0.5 is correct.
