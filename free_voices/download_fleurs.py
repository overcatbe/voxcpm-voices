import os, sys
from huggingface_hub import hf_hub_download

OUT = os.path.join((os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")), "free_voices", "fleurs")
LANGS = {"de": "de_de", "tr": "tr_tr", "ru": "ru_ru", "ar": "ar_eg", "pl": "pl_pl"}

for code, cfg in LANGS.items():
    for fn in (f"data/{cfg}/dev.tsv", f"data/{cfg}/audio/dev.tar.gz"):
        print(f"[{code}] downloading {fn}", flush=True)
        path = hf_hub_download("google/fleurs", fn, repo_type="dataset", local_dir=OUT)
        print(f"[{code}] done {path} ({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
print("ALL DONE", flush=True)
