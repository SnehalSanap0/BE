"""Draw the design diagrams of the final-year project report (architecture, DFD, UML, flowcharts).

The diagrams are drawn with matplotlib (no Graphviz/LaTeX needed) from the real module and class names of this
repository, so they can be regenerated if the design changes:

    python -m scripts.make_report_diagrams

Writes PNG files to ``Final_Year_Project_Report/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from configs.config import ROOT_DIR  # noqa: E402

FIGURES = ROOT_DIR / "Final_Year_Project_Report" / "figures"

BLUE, GREEN, ORANGE, RED, GREY, PURPLE, YELLOW = (
    "#dbe9f6", "#dcefdc", "#fde7c8", "#f8d7d3", "#ececec", "#e6dcf2", "#fff6bf",
)
EDGE = "#34495e"


def _canvas(width: float, height: float, size: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=size)
    axes.set_xlim(0, width)
    axes.set_ylim(0, height)
    axes.axis("off")
    return figure, axes


def box(axes, cx, cy, w, h, text, color=BLUE, fontsize=8.5, bold=False, dashed=False, rounded=True) -> tuple:
    """Draw a labelled box centred on (cx, cy); returns (cx, cy, w, h) for connecting arrows."""
    style = "round,pad=0.02,rounding_size=0.6" if rounded else "square,pad=0.0"
    patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style, facecolor=color, edgecolor=EDGE,
                           linewidth=1.1, linestyle="--" if dashed else "-", zorder=2)
    axes.add_patch(patch)
    axes.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight="bold" if bold else "normal",
              zorder=3, linespacing=1.25)
    return cx, cy, w, h


def arrow(axes, start, end, text="", color=EDGE, dashed=False, rad=0.0, fontsize=7.5, label_offset=(0, 0.9)) -> None:
    """Draw an arrow between two (x, y) points with an optional label at its middle."""
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.1,
                                   linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=1))
    if text:
        axes.text((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1], text,
                  ha="center", va="center", fontsize=fontsize, color="#222", zorder=4,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9})


def right_of(b):  # noqa: ANN001
    return b[0] + b[2] / 2, b[1]


def left_of(b):  # noqa: ANN001
    return b[0] - b[2] / 2, b[1]


def below(b):  # noqa: ANN001
    return b[0], b[1] - b[3] / 2


def above(b):  # noqa: ANN001
    return b[0], b[1] + b[3] / 2


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  {name}")


# --------------------------------------------------------------------------------------------------
def architecture() -> None:
    figure, axes = _canvas(130, 78, (13.5, 8.2))
    axes.text(65, 76, "System architecture", ha="center", fontsize=14, weight="bold")

    # Offline training pipeline
    axes.add_patch(Rectangle((1, 41), 128, 31, facecolor="#f5f9fd", edgecolor="#9db7d3", linestyle="--", zorder=0))
    axes.text(3, 69.5, "A. Training and evaluation pipeline (offline)", fontsize=10, weight="bold", color="#1f4e79")
    sources = box(axes, 11, 56, 18, 17, "Public speech datasets\n\nKathbath, Common\nVoice (Hindi)\nIndicSynth, SEA-Spoof\nWaveFake (English)", GREY)
    ingest = box(axes, 33, 56, 15, 17, "1. Ingestion\n\nbounded HF\nstreaming,\ntranscript\npairing", BLUE)
    prep = box(axes, 51, 56, 15, 17, "2. Pre-\nprocessing\n\nmono, 16 kHz,\ntrim silence,\nnormalise, 4 s", BLUE)
    feat = box(axes, 69, 56, 15, 17, "3. Feature\nextraction\n\nlog-mel\nspectrogram\n224x224 (.npy)", BLUE)
    split = box(axes, 87, 56, 15, 17, "4. Grouped\nsplit\n\nno transcript\nshared across\ntrain/val/test", BLUE)
    train = box(axes, 105, 56, 15, 17, "5. Training\n\nEfficientNet-B0\nAdamW, cosine,\nearly stopping", ORANGE)
    evalb = box(axes, 121, 56, 13, 17, "6. Evaluation\n\nAcc, F1, AUC,\nEER, MCC,\ncross-source", GREEN)
    for a, b in ((sources, ingest), (ingest, prep), (prep, feat), (feat, split), (split, train), (train, evalb)):
        arrow(axes, right_of(a), left_of(b))
    meta = box(axes, 60, 45, 60, 3.6, "metadata.csv  -  single source of truth read and updated by every stage", YELLOW, fontsize=8.5, dashed=True)
    for stage in (ingest, prep, feat, split):
        arrow(axes, below(stage), (stage[0], 46.9), dashed=True)
    ckpt = box(axes, 105, 45, 21, 3.6, "checkpoints/best_model.pt", PURPLE, fontsize=7.5, dashed=True)
    arrow(axes, below(train), above(ckpt), dashed=True)

    # Online forensic pipeline
    axes.add_patch(Rectangle((1, 12), 128, 26, facecolor="#fdf8f2", edgecolor="#d9b48f", linestyle="--", zorder=0))
    axes.text(3, 35.3, "B. Inference and forensic pipeline (online)", fontsize=10, weight="bold", color="#8a4b0f")
    up = box(axes, 11, 24, 16, 14, "Evidence file\n\nWAV, FLAC, MP3,\nOGG/Opus, M4A,\nAMR, ...", GREY)
    val = box(axes, 29, 24, 15, 14, "Validate\n+ SHA-256\n\nguarded ffmpeg\ndecoding", BLUE)
    win = box(axes, 46, 24, 15, 14, "Sliding\nwindows\n\n4 s every 2 s,\nsilence skipped", BLUE)
    mdl = box(axes, 63, 24, 15, 14, "EfficientNet-B0\n\nscore per\nwindow\n(P synthetic)", ORANGE)
    ver = box(axes, 80, 24, 15, 14, "Verdict\n\nlikely genuine /\ninconclusive /\nlikely synthetic", GREEN)
    cam = box(axes, 97, 24, 15, 14, "Grad-CAM\n\nmost suspicious\nwindow\nexplained", GREEN)
    rep = box(axes, 116, 24, 17, 14, "PDF + JSON\nreport\n\nhash-chained\ncustody log", PURPLE)
    for a, b in ((up, val), (val, win), (win, mdl), (mdl, ver), (ver, cam), (cam, rep)):
        arrow(axes, right_of(a), left_of(b))
    arrow(axes, below(ckpt) if False else (105, 43.2), (63, 31), "load trained weights", rad=0.0, label_offset=(7, 1.5))

    # Applications
    axes.text(3, 9, "C. Interfaces", fontsize=10, weight="bold", color="#2c3e50")
    box(axes, 30, 4.5, 24, 5.2, "Gradio web demo\n(compare + forensic tabs)", GREEN)
    box(axes, 65, 4.5, 30, 5.2, "Flask REST API\n/predict   /forensic-report", GREEN)
    box(axes, 100, 4.5, 24, 5.2, "Command line\npython -m forensics.analyze", GREEN)
    for x in (30, 65, 100):
        arrow(axes, (x, 7.2), (x if x != 100 else 98, 12.0), dashed=True)
    _save(figure, "diag_architecture.png")


def model_diagram() -> None:
    figure, axes = _canvas(140, 29, (14, 3.2))
    axes.text(70, 28, "Detection model: log-mel spectrogram + EfficientNet-B0", ha="center", fontsize=13, weight="bold")
    y = 15
    wave = box(axes, 9, y, 15, 14, "Speech\nwaveform\n\n16 kHz mono,\n4 s", GREY)
    mel = box(axes, 28, y, 16, 14, "Log-mel\nspectrogram\n\n128 mel bands\nn_fft 1024\nhop 512", BLUE)
    img = box(axes, 47, y, 14, 14, "Resize to\n224 x 224\n\nz-score\nnormalise\n(1 channel)", BLUE)
    stem = box(axes, 66, y, 17, 14, "EfficientNet-B0\nbackbone\n\nMBConv blocks,\nImageNet-\npretrained", ORANGE)
    pool = box(axes, 85, y, 14, 14, "Global\naverage pool\n\n1280-d\nembedding", ORANGE)
    head = box(axes, 102, y, 14, 14, "Dropout 0.3\n+ Linear\n(1280 to 2)", ORANGE)
    out = box(axes, 120, y, 16, 14, "Softmax\n\nP(genuine)\nP(synthetic)", GREEN)
    for a, b in ((wave, mel), (mel, img), (img, stem), (stem, pool), (pool, head), (head, out)):
        arrow(axes, right_of(a), left_of(b))
    cam = box(axes, 66, 1.6, 64, 3.6, "Grad-CAM on the last conv layer (conv_head): heat-map over time and frequency",
              PURPLE, fontsize=8.5, dashed=True)
    arrow(axes, below(stem), above(cam), dashed=True)
    _save(figure, "diag_model.png")


def pairing_design() -> None:
    figure, axes = _canvas(110, 56, (11.5, 6.0))
    axes.text(55, 54, "Experimental design: transcript-paired (E2) vs non-paired (E1) training", ha="center",
              fontsize=13, weight="bold")
    corpus = box(axes, 16, 28, 24, 22, "SEA-Spoof corpus\n(Hindi)\n\nfor each transcript T:\ngenuine clip of T\nsynthetic clip of T", GREY)
    # E2
    axes.add_patch(Rectangle((36, 29.5), 72, 22, facecolor="#f2f8fd", edgecolor="#9db7d3", linestyle="--", zorder=0))
    axes.text(38, 49.5, "E2: transcript-paired", fontsize=10.5, weight="bold", color="#1f4e79")
    pair = box(axes, 55, 39, 24, 11, "Match each synthetic clip\nto the genuine clip with\nthe SAME transcript", BLUE)
    grp = box(axes, 85, 39, 26, 11, "Keep both clips of a pair\nin ONE split\n(no transcript leakage)", BLUE)
    arrow(axes, right_of(corpus), left_of(pair), rad=0.1)
    arrow(axes, right_of(pair), left_of(grp))
    # E1
    axes.add_patch(Rectangle((36, 5), 72, 22, facecolor="#fdf6ef", edgecolor="#d9b48f", linestyle="--", zorder=0))
    axes.text(38, 24.5, "E1: non-paired control", fontsize=10.5, weight="bold", color="#8a4b0f")
    ind = box(axes, 55, 14.5, 24, 11, "Sample genuine and\nsynthetic clips\nINDEPENDENTLY", ORANGE)
    strat = box(axes, 85, 14.5, 26, 11, "Ordinary label-stratified\nsplit (no transcript\ninformation)", ORANGE)
    arrow(axes, right_of(corpus), left_of(ind), rad=-0.1)
    arrow(axes, right_of(ind), left_of(strat))
    axes.text(55, 2.2, "Identical in both: sources, 3,000 + 3,000 clips, EfficientNet-B0, optimiser, 50-epoch budget, "
              "3 seeds, external test sets", ha="center", fontsize=8.5, style="italic")
    _save(figure, "diag_pairing_design.png")


# --------------------------------------------------------------------------------------------------
def dfd_level0() -> None:
    figure, axes = _canvas(110, 60, (11, 6.0))
    axes.text(55, 58, "Data flow diagram - Level 0 (context)", ha="center", fontsize=13, weight="bold")
    system = Ellipse((55, 30), 34, 22, facecolor=BLUE, edgecolor=EDGE, linewidth=1.3, zorder=2)
    axes.add_patch(system)
    axes.text(55, 30, "Audio deepfake\ndetection and\nforensic analysis\nsystem", ha="center", va="center",
              fontsize=10, weight="bold", zorder=3)
    examiner = box(axes, 12, 46, 20, 8, "Examiner /\ninvestigator", GREEN, rounded=False)
    researcher = box(axes, 12, 14, 20, 8, "Researcher /\ndeveloper", GREEN, rounded=False)
    hf = box(axes, 98, 46, 20, 8, "Public speech\ndatasets (Hugging Face)", GREEN, rounded=False)
    expert = box(axes, 98, 14, 20, 8, "Court / forensic\nexpert (reader)", GREEN, rounded=False)
    arrow(axes, right_of(examiner), (41, 36), "evidence audio,\ncase ID, examiner")
    arrow(axes, (42, 33), (22, 43), "verdict, scores,\nGrad-CAM, PDF report", rad=-0.2, label_offset=(-1, -2.5))
    arrow(axes, right_of(researcher), (42, 24), "config, training\nrequests")
    arrow(axes, (44, 27), (22, 17), "metrics, figures,\ncheckpoints", rad=0.2, label_offset=(1, 2.5))
    arrow(axes, left_of(hf), (69, 36), "speech clips,\ntranscripts")
    arrow(axes, (69, 24), left_of(expert), "report + custody log\n(read-only copy)", label_offset=(0, -2))
    _save(figure, "diag_dfd0.png")


def dfd_level1() -> None:
    figure, axes = _canvas(130, 78, (13.5, 8.0))
    axes.text(65, 76, "Data flow diagram - Level 1", ha="center", fontsize=13, weight="bold")

    def process(cx, cy, w, h, label, color):  # noqa: ANN001
        axes.add_patch(Ellipse((cx, cy), w, h, facecolor=color, edgecolor=EDGE, zorder=2))
        axes.text(cx, cy, label, ha="center", va="center", fontsize=8.5, weight="bold", zorder=3)

    process(16, 56, 22, 13, "1.0 Ingest\ndatasets", BLUE)
    process(44, 56, 22, 13, "2.0 Pre-process\naudio", BLUE)
    process(72, 56, 22, 13, "3.0 Extract\nfeatures + split", BLUE)
    process(104, 56, 22, 13, "4.0 Train and\nevaluate model", ORANGE)
    process(48, 14, 28, 14, "5.0 Analyse evidence\n(windows, score,\nGrad-CAM, verdict)", GREEN)
    process(96, 14, 26, 14, "6.0 Generate report\nand custody entry", PURPLE)

    hf = box(axes, 16, 72, 22, 5, "Public datasets", GREEN, rounded=False)
    d1 = box(axes, 30, 41, 18, 4.5, "D1  raw audio", YELLOW, rounded=False)
    d2 = box(axes, 62, 41, 26, 4.5, "D2  metadata.csv + features", YELLOW, rounded=False)
    d3 = box(axes, 104, 41, 24, 4.5, "D3  checkpoints + metrics", YELLOW, rounded=False)
    ev = box(axes, 10, 14, 16, 7, "Examiner", GREEN, rounded=False)
    d4 = box(axes, 96, 2.6, 30, 4.2, "D4  custody log + reports", YELLOW, rounded=False)

    arrow(axes, below(hf), (16, 62.5), "clips + text", label_offset=(6, 0))
    arrow(axes, (27, 56), (33, 56))
    arrow(axes, (55, 56), (61, 56))
    arrow(axes, (83, 56), (93, 56), "feature maps,\nsplits", label_offset=(0, 4))
    arrow(axes, (16, 49.5), (28, 43.3), "raw clips", label_offset=(-5, 0))
    arrow(axes, (33, 43.3), (42, 49.5), "read", label_offset=(-1, -2.2))
    arrow(axes, (46, 49.5), (56, 43.3), "processed", label_offset=(7, 1.6))
    arrow(axes, (70, 49.5), (66, 43.3), "features", label_offset=(6, 0))
    arrow(axes, (75, 42), (97, 50), "read splits", label_offset=(0, -2.5))
    arrow(axes, (104, 49.5), (104, 43.3), "weights", label_offset=(6, 0))
    arrow(axes, (100, 38.7), (56, 20.5), "load weights", rad=0.1, label_offset=(4, 3))
    arrow(axes, right_of(ev), (34, 14), "evidence audio,\ncase ID", label_offset=(0, 4))
    arrow(axes, (62, 14), (83, 14), "scores, verdict,\nGrad-CAM", label_offset=(0, 4))
    arrow(axes, (96, 7), (96, 4.8), "PDF, JSON, chain entry", label_offset=(-16, 0.2))
    arrow(axes, (81, 2.6), (10, 10.4), "report (PDF) back to the examiner", rad=-0.12, label_offset=(0, -4))
    _save(figure, "diag_dfd1.png")


# --------------------------------------------------------------------------------------------------
def use_case() -> None:
    figure, axes = _canvas(140, 72, (14, 7.2))
    axes.text(70, 70, "Use case diagram", ha="center", fontsize=13, weight="bold")
    axes.add_patch(Rectangle((28, 0.5), 84, 65.5, facecolor="#fafcff", edgecolor=EDGE, linewidth=1.3))
    axes.text(70, 63.3, "Audio deepfake detection and forensic system", ha="center", fontsize=9.5, weight="bold")

    def actor(x, y, label):  # noqa: ANN001
        axes.add_patch(plt.Circle((x, y + 6), 1.6, facecolor="white", edgecolor=EDGE, linewidth=1.3))
        axes.plot([x, x], [y + 4.4, y + 0.5], color=EDGE, linewidth=1.3)
        axes.plot([x - 3, x + 3], [y + 3, y + 3], color=EDGE, linewidth=1.3)
        axes.plot([x, x - 2.5], [y + 0.5, y - 2.5], color=EDGE, linewidth=1.3)
        axes.plot([x, x + 2.5], [y + 0.5, y - 2.5], color=EDGE, linewidth=1.3)
        axes.text(x, y - 5.5, label, ha="center", fontsize=9, weight="bold")
        return x, y + 3

    def case(x, y, label, color):  # noqa: ANN001
        axes.add_patch(Ellipse((x, y), 36, 6.4, facecolor=color, edgecolor=EDGE, zorder=2))
        axes.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3)

    examiner = actor(12, 34, "Examiner /\ninvestigator")
    researcher = actor(128, 34, "Researcher /\ndeveloper")
    left_x, right_x = 52, 88
    for index, label in enumerate(["Upload evidence audio", "Analyse whole recording",
                                   "View verdict and known error rates", "View Grad-CAM explanation",
                                   "Download PDF / JSON report", "Verify custody log"]):
        y = 55 - index * 8.6
        case(left_x, y, label, GREEN)
        axes.plot([examiner[0] + 3, left_x - 18], [examiner[1], y], color=EDGE, linewidth=0.9)
    for index, label in enumerate(["Ingest and pair datasets", "Train the detector", "Evaluate cross-source",
                                   "Run codec robustness check", "Generate report figures"]):
        y = 55 - index * 8.6
        case(right_x, y, label, ORANGE)
        axes.plot([researcher[0] - 3, right_x + 18], [researcher[1], y], color=EDGE, linewidth=0.9)
    case(70, 4.2, "Compare real vs fake clips", BLUE)
    axes.plot([examiner[0] + 3, 52], [examiner[1], 4.2], color=EDGE, linewidth=0.9)
    axes.plot([researcher[0] - 3, 88], [researcher[1], 4.2], color=EDGE, linewidth=0.9)
    _save(figure, "diag_usecase.png")


def class_diagram() -> None:
    figure, axes = _canvas(150, 92, (15.5, 9.4))
    axes.text(75, 90, "Class diagram (principal classes)", ha="center", fontsize=13, weight="bold")

    def klass(cx, top, w, name, attrs, methods, color=BLUE):  # noqa: ANN001
        height = 4.2 + 2.5 * (len(attrs) + len(methods)) + 1.5
        axes.add_patch(Rectangle((cx - w / 2, top - height), w, height, facecolor="white", edgecolor=EDGE,
                                 linewidth=1.2, zorder=2))
        axes.add_patch(Rectangle((cx - w / 2, top - 4.2), w, 4.2, facecolor=color, edgecolor=EDGE, linewidth=1.2,
                                 zorder=3))
        axes.text(cx, top - 2.1, name, ha="center", va="center", fontsize=8.5, weight="bold", zorder=4)
        y = top - 4.2 - 1.8
        for line in attrs:
            axes.text(cx - w / 2 + 1, y, line, fontsize=7, va="center", zorder=4)
            y -= 2.5
        axes.plot([cx - w / 2, cx + w / 2], [y + 1.25, y + 1.25], color=EDGE, linewidth=0.8, zorder=4)
        y -= 0.3
        for line in methods:
            axes.text(cx - w / 2 + 1, y, line, fontsize=7, va="center", zorder=4)
            y -= 2.5
        return {"cx": cx, "top": top, "w": w, "h": height}

    def point(k, side):  # noqa: ANN001
        return {"t": (k["cx"], k["top"]), "b": (k["cx"], k["top"] - k["h"]),
                "l": (k["cx"] - k["w"] / 2, k["top"] - k["h"] / 2),
                "r": (k["cx"] + k["w"] / 2, k["top"] - k["h"] / 2)}[side]

    def edge(a, b, text, sides, dashed=False, rad=0.0, label_at=None):  # noqa: ANN001
        start, end = point(a, sides[0]), point(b, sides[1])
        axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=EDGE, linewidth=1.0,
                                       linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                       zorder=1))
        x, y = label_at or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.2)
        axes.text(x, y, text, fontsize=7, ha="center", va="center", zorder=5,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

    validator = klass(16, 84, 24, "AudioValidator", ["supported_formats"], ["validate(path)", "is_supported_format()"], GREEN)
    loader = klass(16, 62, 24, "audio_loader", ["FFMPEG_DEMUXERS"], ["load_audio(path)", "read_audio_info(path)"], GREEN)
    trainer = klass(16, 40, 24, "Trainer", ["model, optimizer", "scheduler, callbacks"], ["fit(num_epochs)"], ORANGE)
    ckpt = klass(16, 20, 24, "CheckpointManager", ["directory"], ["save(...)", "load(path)", "load_latest()"], YELLOW)

    extractor = klass(48, 84, 26, "FeatureExtractor", ["image_size", "feature_type"],
                      ["extract_file(in, out)", "extract_array(audio, sr)"])
    cam = klass(48, 62, 26, "GradCAMExplainer", ["model", "target_layer"], ["generate(tensor, class)"], PURPLE)
    model = klass(48, 42, 26, "EfficientNetClassifier", ["backbone: timm B0", "classifier: Dropout+Linear"],
                  ["forward(features)"], ORANGE)

    service = klass(86, 84, 30, "InferenceService", ["model", "feature_extractor", "explainer", "checkpoint_sha256"],
                    ["predict(path)", "explain_waveform(audio, sr)", "fake_scores(windows)"], ORANGE)
    segments = klass(86, 50, 30, "SegmentAnalysis", ["segments: tuple[Segment]", "window_seconds", "hop_seconds"],
                     ["scored", "most_suspicious"], BLUE)
    evaluator = klass(86, 26, 30, "Evaluator", ["model, dataloader"], ["evaluate()", "save_confusion_matrix()"], GREEN)

    analyse = klass(124, 84, 32, "forensics.analyze", ["thresholds (configured)"],
                    ["analyze_recording(path, svc)", "analyze_evidence(path, case, svc)", "validated_error_rates(svc)"],
                    GREEN)
    findings = klass(124, 58, 32, "ForensicFindings", ["evidence: EvidenceInfo", "verdict, fake_score",
                                                       "validated: bool", "error_rates", "warnings"],
                     ["to_dict()", "integrity_verified"], PURPLE)
    custody = klass(124, 28, 32, "CustodyLog", ["log_path"], ["append(action, details)", "verify()"], YELLOW)

    edge(validator, loader, "uses", ("b", "t"))
    edge(trainer, model, "trains", ("r", "l"))
    edge(trainer, ckpt, "saves", ("b", "t"))
    edge(cam, model, "explains", ("b", "t"))
    edge(service, extractor, "uses", ("l", "r"), label_at=(67, 80.5))
    edge(service, cam, "owns", ("l", "r"), label_at=(65, 66))
    edge(service, model, "owns", ("l", "r"), label_at=(65.5, 46))
    edge(evaluator, model, "evaluates", ("l", "r"), dashed=True, label_at=(66, 29))
    edge(analyse, service, "calls", ("l", "r"), label_at=(105, 75))
    edge(analyse, findings, "builds", ("b", "t"))
    edge(findings, segments, "contains", ("l", "r"), label_at=(105, 45))
    edge(analyse, custody, "appends", ("r", "r"), dashed=True, rad=-0.35, label_at=(145, 55))
    axes.text(75, 3, "forensics.analyze also uses AudioValidator and audio_loader to validate and decode the evidence "
              "(edges omitted for clarity).", ha="center", fontsize=8, style="italic")
    _save(figure, "diag_class.png")


# --------------------------------------------------------------------------------------------------
def sequence() -> None:
    figure, axes = _canvas(150, 80, (15.5, 8.4))
    axes.text(75, 78, "Sequence diagram - forensic analysis of one recording", ha="center", fontsize=13, weight="bold")
    actors = [("Examiner", 12), ("Gradio UI /\nFlask API", 34), ("forensics.\nanalyze", 58), ("audio_loader +\nAudioValidator", 83),
              ("Inference\nService", 108), ("Report writer +\nCustodyLog", 133)]
    for label, x in actors:
        box(axes, x, 71, 18, 6, label, BLUE, fontsize=8, bold=True)
        axes.plot([x, x], [68, 3], color="#999", linewidth=0.9, linestyle="--", zorder=0)
    steps = [
        (12, 34, 64, "upload evidence, case ID, examiner"),
        (34, 58, 59, "analyze_evidence(file, case, examiner, service)"),
        (58, 83, 54, "describe_evidence: SHA-256, size, format"),
        (58, 83, 49, "validate + decode (guarded ffmpeg)"),
        (83, 58, 46, "mono waveform"),
        (58, 108, 41, "fake_scores(windows)  [4 s every 2 s]"),
        (108, 58, 37, "probability synthetic per window"),
        (58, 108, 33, "explain_waveform(most suspicious window)"),
        (108, 58, 29, "spectrogram + Grad-CAM + peak region"),
        (58, 58, 24.5, "classify windows; verdict = worst window; look up validated error rates"),
        (58, 83, 19.5, "re-hash evidence (integrity check)"),
        (58, 133, 15.5, "write PDF + JSON; append hash-chained entry"),
        (133, 58, 11.5, "report paths, entry hash"),
        (58, 34, 8, "ForensicAnalysis(findings, files)"),
        (34, 12, 4.5, "verdict, scores, timeline, Grad-CAM, PDF"),
    ]
    for start, end, y, text in steps:
        if start == end:
            axes.text(start + 2, y, text, fontsize=7.5, va="center", ha="left",
                      bbox={"facecolor": YELLOW, "edgecolor": EDGE, "pad": 2.0, "boxstyle": "round,pad=0.35"})
            continue
        axes.annotate("", xy=(end, y), xytext=(start, y),
                      arrowprops={"arrowstyle": "-|>", "color": EDGE, "linewidth": 1.0,
                                  "linestyle": "--" if end < start else "-"})
        axes.text((start + end) / 2, y + 1.15, text, ha="center", fontsize=7.5,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
    _save(figure, "diag_sequence.png")


def flowchart() -> None:
    figure, axes = _canvas(130, 116, (11.0, 10.6))
    axes.set_ylim(-4, 116)
    axes.text(65, 113, "Flowchart - forensic analysis and verdict logic", ha="center", fontsize=13, weight="bold")

    def diamond(cx, cy, w, h, text):  # noqa: ANN001
        axes.add_patch(plt.Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                                   facecolor=YELLOW, edgecolor=EDGE, linewidth=1.1, zorder=2))
        axes.text(cx, cy, text, ha="center", va="center", fontsize=8.5, zorder=3)
        return cx, cy, w, h

    cx, w = 42, 54
    start = box(axes, cx, 108, w, 5, "Start: evidence file, case ID, examiner", GREEN)
    d1 = diamond(cx, 97, 44, 11, "Case ID and examiner\nboth given?")
    hashb = box(axes, cx, 85, w, 5, "Compute SHA-256 and read file metadata", BLUE)
    d2 = diamond(cx, 74, 44, 11, "Valid, decodable\naudio?")
    win = box(axes, cx, 62.5, w, 6, "Convert to mono 16 kHz; split into 4 s windows\n(skip mostly silent windows)", BLUE)
    scr = box(axes, cx, 52, w, 5, "Score every window with EfficientNet-B0", ORANGE)
    d3 = diamond(cx, 41, 44, 11, "Model has measured\nerror rates?")
    notval = box(axes, cx, 28, w, 6, "NO: every window is INCONCLUSIVE\n(model not validated)", RED)
    verdict = box(axes, cx, 17, w, 5, "Verdict follows the most suspicious window", BLUE)
    warn = box(axes, cx, 8.5, w, 6, "Add warnings: lossy codec, low sample rate,\nmany windows, little speech", BLUE)
    rep = box(axes, cx, 1.5, w, 4.5, "Re-hash; write PDF + JSON; append custody entry", PURPLE)
    thr = box(axes, 102, 41, 42, 7, "YES: thresholds\ngenuine < 0.05, synthetic > 0.95", GREEN)
    rej1 = box(axes, 102, 97, 40, 6, "Reject: both fields required", RED)
    rej2 = box(axes, 102, 74, 40, 6, "Reject with explanation", RED)

    arrow(axes, below(start), above(d1))
    arrow(axes, below(d1), above(hashb), "yes", label_offset=(3, 0))
    arrow(axes, below(hashb), above(d2))
    arrow(axes, below(d2), above(win), "yes", label_offset=(3, 0))
    arrow(axes, below(win), above(scr))
    arrow(axes, below(scr), above(d3))
    arrow(axes, (cx + 22, 97), left_of(rej1), "no", label_offset=(0, 1.2))
    arrow(axes, (cx + 22, 74), left_of(rej2), "no", label_offset=(0, 1.2))
    arrow(axes, (cx + 22, 41), left_of(thr), "yes", label_offset=(0, 1.2))
    arrow(axes, below(d3), above(notval), "no", label_offset=(3, 0))
    arrow(axes, below(notval), above(verdict))
    arrow(axes, below(thr), (cx + w / 2, 17.5), "")
    arrow(axes, below(verdict), above(warn))
    arrow(axes, below(warn), above(rep))
    _save(figure, "diag_flowchart.png")


def main() -> None:
    for title, function in (("Architecture", architecture), ("Model", model_diagram), ("Pairing design", pairing_design),
                            ("DFD level 0", dfd_level0), ("DFD level 1", dfd_level1), ("Use case", use_case),
                            ("Class diagram", class_diagram), ("Sequence", sequence), ("Flowchart", flowchart)):
        print(title)
        function()
    print(f"Done. Diagrams in {FIGURES}")


if __name__ == "__main__":
    main()
