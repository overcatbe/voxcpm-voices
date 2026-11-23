"""Pick clean reference clips from downloaded FLEURS splits.

usage: screen_fleurs.py SPEC [SPEC ...]
  SPEC = code[:split[:genders]]   e.g.  ru   de:train:female   pl:dev:female,male
  split defaults to dev, genders default to both.

For each spec: unpack the split's tar.gz, keep clips of 8-16 s, drop clipped or
noisy-start ones, rank by a rough SNR proxy, then pick PER_GENDER clips per gender
whose median pitch differs (FLEURS has no speaker ids, so this is a crude guard
against picking two clips of the same person). Picks go to
candidates/fleurs_<code>_<split>_<gender>_<n>.wav with a .reference.txt transcript.
"""
import csv
import os
import sys
import tarfile

import librosa
import numpy as np
import soundfile as sf

BASE = os.path.join((os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")), "free_voices")
CFG = {"de": "de_de", "tr": "tr_tr", "ru": "ru_ru", "ar": "ar_eg", "pl": "pl_pl"}
PER_GENDER = 2
MIN_S, MAX_S = 8.0, 16.0
SHORTLIST = 20        # best-SNR clips per gender that get a pitch estimate
MIN_F0_DIFF = 0.08    # picks must differ by at least 8 % in median pitch


def frame_db(x, sr, ms=25):
    n = int(sr * ms / 1000)
    m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def score(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    db = frame_db(x, sr)
    return {
        "dur": len(x) / sr,
        "peak": float(np.abs(x).max()),
        "snr": float(np.percentile(db, 95) - np.percentile(db, 10)),  # bigger = cleaner
        "lead": float(db[:8].mean()),  # first 200 ms should be quiet, not speech
        "sr": sr,
    }


def median_f0(path):
    x, sr = librosa.load(path, sr=16000)
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=sr)
    voiced = f0[(f0 > 62) & (f0 < 395)]
    return float(np.median(voiced)) if len(voiced) else 0.0


def run_spec(spec):
    parts = spec.split(":")
    code = parts[0]
    split = parts[1] if len(parts) > 1 and parts[1] else "dev"
    genders = parts[2].split(",") if len(parts) > 2 and parts[2] else ["female", "male"]
    cfg = CFG[code]
    root = os.path.join(BASE, "fleurs", "data", cfg)
    ex_dir = os.path.join(BASE, "fleurs", "extracted", code)
    if not os.path.isdir(os.path.join(ex_dir, split)):
        os.makedirs(ex_dir, exist_ok=True)
        with tarfile.open(os.path.join(root, "audio", f"{split}.tar.gz")) as t:
            t.extractall(ex_dir, filter="data")

    rows = []
    with open(os.path.join(root, f"{split}.tsv"), encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) >= 7 and r[-1].strip().lower() in genders:
                rows.append({"file": r[1], "text": r[2], "gender": r[-1].strip().lower()})

    scored = []
    for r in rows:
        p = os.path.join(ex_dir, split, r["file"])
        if not os.path.exists(p):
            continue
        s = score(p)
        if MIN_S <= s["dur"] <= MAX_S and s["peak"] < 0.99 and s["lead"] < -45:
            scored.append({**r, **s, "path": p})
    print(f"[{code}:{split}] {len(rows)} rows for {genders}, {len(scored)} pass basic checks")

    out_dir = os.path.join(BASE, "candidates")
    os.makedirs(out_dir, exist_ok=True)
    for gender in genders:
        pool = sorted((s for s in scored if s["gender"] == gender), key=lambda s: -s["snr"])[:SHORTLIST]
        for s in pool:
            s["f0"] = median_f0(s["path"])
        picked, seen_text = [], set()
        for s in pool:
            if s["text"] in seen_text or s["f0"] == 0.0:
                continue
            if any(abs(s["f0"] - p["f0"]) / p["f0"] < MIN_F0_DIFF for p in picked):
                continue
            seen_text.add(s["text"])
            picked.append(s)
            if len(picked) == PER_GENDER:
                break
        if len(picked) < PER_GENDER:
            print(f"   WARNING: only {len(picked)} distinct-pitch {gender} clip(s) found")
        for i, s in enumerate(picked, 1):
            name = f"fleurs_{code}_{split}_{gender}_{i}"
            data, sr = sf.read(s["path"], dtype="float32")
            sf.write(os.path.join(out_dir, name + ".wav"), data, sr)
            with open(os.path.join(out_dir, name + ".reference.txt"), "w", encoding="utf-8") as f:
                f.write(s["text"] + "\n")
            print(f"   {name}: {s['dur']:.1f}s snr~{s['snr']:.0f}dB f0~{s['f0']:.0f}Hz peak={s['peak']:.2f} sr={s['sr']}")


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        run_spec(spec)
    # CC BY 4.0 asks for credit wherever these voices are used: keep the attribution next to them
    import shutil
    os.makedirs(os.path.join(BASE, "candidates"), exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FLEURS_ATTRIBUTION.txt"),
                os.path.join(BASE, "candidates"))
