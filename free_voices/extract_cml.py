"""Cut short clean reference clips per speaker from downloaded CML-TTS dev shards.
usage: extract_cml.py de pl   (shards in free_voices/cml/<language>/dev-*.parquet)"""
import glob, io, os, sys
import numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq

BASE = os.path.join((os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")), "free_voices"); OUT = BASE + "/candidates"
LANGDIR = {"de": "german", "pl": "polish"}
WANT = {"de": 4, "pl": 4}

def snr(x, sr):
    n = int(sr * .025); m = len(x) // n
    db = 20 * np.log10(np.sqrt((x[:m*n].reshape(m, n) ** 2).mean(axis=1)) + 1e-9)
    return float(np.percentile(db, 95) - np.percentile(db, 10))

def f0(x, sr):
    y = librosa.resample(x, orig_sr=sr, target_sr=16000)
    v = librosa.yin(y, fmin=60, fmax=400, sr=16000); v = v[(v > 62) & (v < 395)]
    return float(np.median(v)) if len(v) else 0.0

for lang in sys.argv[1:]:
    t = pq.read_table(glob.glob(f"{BASE}/cml/{LANGDIR[lang]}/dev-*.parquet")[0])
    df = t.select([c for c in t.column_names if c != "audio"]).to_pandas()
    audio = t.column("audio").to_pylist()
    made = []
    for spk, g in df.groupby("speaker_id"):
        g = g[(g["duration"] >= 3) & (g["duration"] <= 14)]
        if g["duration"].sum() < 30: continue
        clips, texts, total = [], [], 0.0
        for i in g.sort_values("levenshtein").head(12).index:
            x, sr = sf.read(io.BytesIO(audio[i]["bytes"]), dtype="float32")
            if x.ndim > 1: x = x.mean(axis=1)
            if np.abs(x).max() >= 0.99 or snr(x, sr) < 25: continue
            clips.append(x); total += len(x) / sr
            if isinstance(df.loc[i, "text"], str): texts.append(df.loc[i, "text"])   # some rows have no transcript
            if total >= 22 or len(clips) >= 6: break
        if total < 12: continue
        gap = np.zeros(int(sr * .3), dtype="float32")
        made.append((f0(np.concatenate([c for x in clips for c in (x, gap)][:-1]), sr), spk,
                     np.concatenate([c for x in clips for c in (x, gap)][:-1]), sr, texts, total, len(clips)))
    made.sort(key=lambda m: m[0])
    k = WANT[lang]
    pick = made[:k // 2] + made[-(k - k // 2):] if len(made) > k else made
    for fz, spk, ref, sr, texts, total, n in pick:
        name = f"cml_{lang}_{spk}"
        sf.write(f"{OUT}/{name}.wav", ref, sr)
        open(f"{OUT}/{name}.reference.txt", "w", encoding="utf-8").write(" ".join(texts) + "\n")
        print(f"{name:16s} {total:5.1f}s from {n} clips, sr={sr}, median pitch {fz:3.0f} Hz ({'lower' if fz < 165 else 'higher'} voice)")
    print(f"[{lang}] {df['speaker_id'].nunique()} speakers in shard, {len(made)} had enough clean material, {len(pick)} picked")

# CC BY 4.0 asks for credit wherever these voices are used: keep the attribution next to them
import shutil
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "CML_TTS_ATTRIBUTION.txt"), OUT)
