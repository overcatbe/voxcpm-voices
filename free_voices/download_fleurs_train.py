import os
from huggingface_hub import hf_hub_download

OUT = os.path.join((os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")), "free_voices", "fleurs")
for cfg in ("de_de", "tr_tr", "ar_eg"):
    fn = f"data/{cfg}/audio/train.tar.gz"
    print(f"downloading {fn}", flush=True)
    p = hf_hub_download("google/fleurs", fn, repo_type="dataset", local_dir=OUT)
    print(f"done {p} ({os.path.getsize(p)/1e6:.0f} MB)", flush=True)
print("ALL DONE", flush=True)
