"""Build ~30 s reference voices from CML-TTS through Hugging Face's dataset viewer API.
Only the clips we pick are downloaded (about 0.7 MB each), no dataset shards.
usage: fetch_cml_api.py de pl        (build both lower and higher voices)
       fetch_cml_api.py de:higher     (only keep voices with a higher pitch; also de:lower)"""
import glob, io, json, os, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
import numpy as np, soundfile as sf, librosa

BASE = "https://datasets-server.huggingface.co"
OUT = os.path.join((os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")), "free_voices", "candidates")
CONFIG = {"de": "german", "pl": "polish"}
SPLITS = ("dev", "test")
CANDIDATE_SPEAKERS = 8      # speakers per language we actually download clips for
KEEP = 4                    # final voices per language: half lower, half higher pitch
TARGET_MIN, TARGET_MAX = 28.0, 34.0

def api(path, tries=8):
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"{BASE}{path}", timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if i == tries - 1: raise
            wait = int(e.headers.get("Retry-After", 0) or 0) if e.code == 429 else 0
            wait = max(wait, 5 * (i + 1))          # unauthenticated API is rate limited: back off
            print(f"   HTTP {e.code}, waiting {wait}s", flush=True); time.sleep(wait)
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 * (i + 1))

def fetch_audio(url):
    with urllib.request.urlopen(url, timeout=90) as r: raw = r.read()
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:  # format libsndfile cannot read: let ffmpeg convert
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"], input=raw, capture_output=True, check=True)
        x, sr = sf.read(io.BytesIO(p.stdout), dtype="float32")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr, len(raw)

def snr(x, sr):
    n = int(sr * .025); m = len(x) // n
    db = 20 * np.log10(np.sqrt((x[:m*n].reshape(m, n) ** 2).mean(axis=1)) + 1e-9)
    return float(np.percentile(db, 95) - np.percentile(db, 10))

def f0(x, sr):
    y = librosa.resample(x, orig_sr=sr, target_sr=16000)
    v = librosa.yin(y, fmin=60, fmax=400, sr=16000); v = v[(v > 62) & (v < 395)]
    return float(np.median(v)) if len(v) else 0.0

total_bytes = 0
for spec in sys.argv[1:]:
    lang, _, want = spec.partition(":")
    n_cand = CANDIDATE_SPEAKERS + (4 if want else 0)
    have = {os.path.basename(p).split("_")[2].split(".")[0] for p in glob.glob(f"{OUT}/cml_{lang}_*.wav")}
    clips = {}                                             # speaker -> list of row dicts
    for split in SPLITS:
        off, total = 0, None
        while total is None or off < total:
            d = api(f"/rows?dataset=ylacombe/cml-tts&config={CONFIG[lang]}&split={split}&offset={off}&length=100")
            total = d["num_rows_total"]
            for r in d["rows"]:
                row = r["row"]
                spk = str(row["speaker_id"])
                if spk in have or not (3 <= row["duration"] <= 14): continue
                clips.setdefault(spk, []).append({"url": row["audio"][0]["src"], "dur": row["duration"], "lev": row["levenshtein"],
                                                  "text": row["text"] if isinstance(row["text"], str) else None, "split": split})
            off += 100
            enough = sum(1 for c in clips.values() if sum(x["dur"] for x in c) >= 45)
            if enough >= n_cand + 4: break                 # enough speakers seen, stop scanning early
            time.sleep(1.0)                                # be gentle with the API
        if sum(1 for c in clips.values() if sum(x["dur"] for x in c) >= 45) >= n_cand + 4: break
    ranked = sorted(clips, key=lambda s: -len(clips[s]))
    ranked = [s for s in ranked if sum(c["dur"] for c in clips[s]) >= 45][:n_cand]
    print(f"[{lang}] {len(clips)} new speakers found in dev+test; downloading clips for the {len(ranked)} with most material", flush=True)
    built = []
    for spk in ranked:
        parts, texts, tot = [], [], 0.0
        for c in sorted(clips[spk], key=lambda c: c["lev"])[:16]:
            if tot >= TARGET_MIN: break
            if tot + c["dur"] > TARGET_MAX: continue
            try: x, sr, nbytes = fetch_audio(c["url"])
            except Exception as e:
                print(f"   clip download failed ({type(e).__name__}), skipping", flush=True); continue
            total_bytes += nbytes
            if np.abs(x).max() >= 0.99 or snr(x, sr) < 25: continue
            parts.append(x); tot += len(x) / sr
            if c["text"]: texts.append(c["text"])
        if tot < 22: print(f"   speaker {spk}: only {tot:.0f}s of clean material, skipped", flush=True); continue
        gap = np.zeros(int(sr * .3), dtype="float32")
        ref = np.concatenate([p for x in parts for p in (x, gap)][:-1])
        built.append((f0(ref, sr), spk, ref, sr, texts, tot, len(parts)))
    built.sort(key=lambda b: b[0])
    if want == "higher": pick = [b for b in built if b[0] >= 165][-KEEP // 2:]
    elif want == "lower": pick = [b for b in built if b[0] < 165][:KEEP // 2]
    else: pick = built[:KEEP // 2] + built[-(KEEP - KEEP // 2):] if len(built) > KEEP else built
    for fz, spk, ref, sr, texts, tot, n in pick:
        name = f"cml_{lang}_{spk}"
        sf.write(f"{OUT}/{name}.wav", ref, sr)
        open(f"{OUT}/{name}.reference.txt", "w", encoding="utf-8").write(" ".join(texts) + "\n")
        print(f"   {name:16s} {len(ref)/sr:5.1f}s from {n} clips, sr={sr}, median pitch {fz:3.0f} Hz ({'lower' if fz < 165 else 'higher'} voice)", flush=True)
    print(f"[{lang}] built {len(built)} candidates, kept {len(pick)}", flush=True)
print(f"total audio downloaded: {total_bytes/1e6:.0f} MB")

# CC BY 4.0 asks for credit wherever these voices are used: keep the attribution next to them
import shutil
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "CML_TTS_ATTRIBUTION.txt"), OUT)
