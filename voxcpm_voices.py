"""voxcpm-voices: pick, tune and save VoxCPM2 voices in the browser, with a small API.

Pick language + voice, tune the settings, generate a preview (saved as preview.mp3),
save the profile. All profiles go into ONE settings file (profiles/voice_profiles.json).
Other programs use the voices through two API calls, `voices` and `speak` (see README.md).

Voices offered: VCTK clips you put into vctk/ (CC BY 4.0), the FLEURS and CML-TTS candidates made by
the scripts in free_voices/ (CC BY 4.0, see the *_ATTRIBUTION.txt files), and voices added through
the UI from your own recordings (profiles/custom_voices/, each with a required source/consent note).
Everything lives in $VOXCPM_VOICES_HOME (default ~/.local/share/voxcpm-voices).
"""
import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import glob
import json
import os
import re
import gc
import io
import shutil
import subprocess
from datetime import datetime, timezone

MODEL_ID = "openbmb/VoxCPM2"
HOME_DIR = os.environ.get("VOXCPM_VOICES_HOME") or os.path.expanduser("~/.local/share/voxcpm-voices")
VOICES_DIR = os.path.join(HOME_DIR, "vctk")                      # optional: vctk_*.wav clips
CANDIDATES_DIR = os.path.join(HOME_DIR, "free_voices", "candidates")  # made by free_voices/*.py
PROFILES_DIR = os.path.join(HOME_DIR, "profiles")
PORT = int(os.environ.get("VOXCPM_VOICES_PORT", "7872"))
SETTINGS_FILE = os.path.join(PROFILES_DIR, "voice_profiles.json")
CUSTOM_DIR = os.path.join(PROFILES_DIR, "custom_voices")  # voices added through the UI
FLAGS_FILE = os.path.join(PROFILES_DIR, "voice_flags.json")  # voices marked BAD
META_FILE = os.path.join(PROFILES_DIR, "voice_meta.json")  # display names + gender per voice
AVATARS_DIR = os.path.join(PROFILES_DIR, "avatars")  # one square PNG per voice, <technical voice id>.png
AVATAR_SIZE = 512
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # avatar images
TESTS_DIR = os.path.join(os.path.dirname(CANDIDATES_DIR), "tests")  # test outputs of the candidates
os.makedirs(CUSTOM_DIR, exist_ok=True)

# VoxCPM2 has no language parameter, it infers from the text, so the code selects the
# sample sentence and is stored in the profile. Only languages it covers from our list.
SAMPLES = {
    "EN": "Hello! This is how this voice sounds. Do you like it?",
    "DE": "Hallo! So klingt diese Stimme. Gefällt sie dir?",
    "TR": "Merhaba! Bu ses böyle duyuluyor. Beğendiniz mi?",
    "RU": "Привет! Вот так звучит этот голос. Вам нравится?",
    "AR": "مرحبًا! هكذا يبدو هذا الصوت. هل يعجبك؟",
    "PL": "Cześć! Tak brzmi ten głos. Podoba ci się?",
}
STATUSES = ["untested", "approved", "rejected"]

# Mood presets. VoxCPM2 takes a mood as a short plain-language instruction in brackets glued to
# the front of the text. The tool adds the brackets itself, so people only pick or type the words.
# The wording is ours and untested per voice/language: listen before relying on a preset.
MOODS = {
    "(none)": "",
    "Neutral narrator": "clear, neutral narration",
    "Friendly": "friendly, warm tone",
    "Encouraging": "encouraging, supportive tone",
    "Calm and reassuring": "calm, reassuring tone, slightly slower",
    "Cheerful": "cheerful, upbeat tone",
    "Professional": "serious, professional tone",
    "Energetic": "energetic, motivating tone, slightly faster",
    "Slow and clear": "slow, very clear pronunciation",
    "Urgent warning": "urgent, firm tone",
    "Sad, sympathetic": "sad, sympathetic tone, slower",
}
MOOD_CUSTOM = "(custom)"


def clean_style(style):
    """Style words without surrounding brackets or spaces."""
    style = (style or "").strip()
    if style.startswith("(") and style.endswith(")"):
        style = style[1:-1].strip()
    return style


def mood_for(style):
    """Which preset name matches these style words ('(none)' if empty, '(custom)' if none does)."""
    style = clean_style(style)
    if not style:
        return "(none)"
    for name, words in MOODS.items():
        if words and words == style:
            return name
    return MOOD_CUSTOM


def discover_voices():
    """Returns (name -> wav path, name -> language). Language 'any' = offered for every language."""
    voices, langs = {}, {}
    for p in sorted(glob.glob(os.path.join(VOICES_DIR, "vctk_*.wav"))):
        name = os.path.splitext(os.path.basename(p))[0]
        voices[name], langs[name] = p, "any"  # English speakers, profiled for all languages
    for pattern in ("fleurs_*.wav", "cml_*.wav"):  # candidates: <source>_<lang>_...
        for p in sorted(glob.glob(os.path.join(CANDIDATES_DIR, pattern))):
            name = os.path.splitext(os.path.basename(p))[0]
            voices[name], langs[name] = p, name.split("_")[1]
    for p in sorted(glob.glob(os.path.join(CUSTOM_DIR, "*.wav"))):
        name = os.path.splitext(os.path.basename(p))[0]
        meta = {}
        try:
            with open(os.path.join(CUSTOM_DIR, name + ".json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        voices[name], langs[name] = p, meta.get("language", "any")
    return voices, langs


VOICES, VOICE_LANG = discover_voices()


def refresh_voices():
    global VOICES, VOICE_LANG
    VOICES, VOICE_LANG = discover_voices()


def read_flags():
    try:
        with open(FLAGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_flags(flags):
    tmp = FLAGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(flags, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, FLAGS_FILE)


def voices_for(lang, show_bad=False):
    # Every voice can be used for every language (cross-lingual cloning); voices recorded in
    # the selected language (and language-neutral ones) are listed first, BAD ones last.
    flags = read_flags()
    names = [v for v in VOICES if show_bad or v not in flags]
    return sorted(names, key=lambda v: (v in flags, VOICE_LANG.get(v, "any") not in ("any", lang.lower())))


def read_meta():
    try:
        with open(META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_meta(meta):
    tmp = META_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, META_FILE)


LANG_NAMES = {"EN": "English", "DE": "German", "TR": "Turkish", "RU": "Russian", "AR": "Arabic", "PL": "Polish"}


def voice_id(voice, meta):
    """The voice id other programs see: the display name if set, else the technical id."""
    return safe_name(meta.get(voice, {}).get("name") or voice)


def voice_label(voice, flags, meta):
    m = meta.get(voice, {})
    bits = [b for b in (LANG_NAMES.get(VOICE_LANG.get(voice, "any").upper(), ""), m.get("gender")) if b]
    label = (m.get("name") or voice) + (f" ({', '.join(bits)})" if bits else "")
    if m.get("name"):
        label += f" [{voice}]"
    return ("BAD | " if voice in flags else "") + label


def voice_choices(lang, show_bad=False):
    flags, meta = read_flags(), read_meta()
    return [(voice_label(v, flags, meta), v) for v in voices_for(lang, show_bad)]


def avatar_path(voice):
    p = os.path.join(AVATARS_DIR, safe_name(voice) + ".png")
    return p if os.path.exists(p) else None


def normalize_avatar(src, dest):
    """Any image -> centered square crop, 512x512 PNG. Re-encoding also drops metadata and any
    non-image payload."""
    from PIL import Image, ImageOps
    if os.path.getsize(src) > MAX_UPLOAD_BYTES:
        raise gr.Error("That image is larger than 10 MB.")
    with Image.open(src) as im:
        if im.width * im.height > 50_000_000:
            raise gr.Error("That image has too many pixels.")
        im = ImageOps.exif_transpose(im).convert("RGBA")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        im.save(dest, "PNG", optimize=True)


# ---- model (loaded on the first preview, so the UI opens instantly) ---------------
_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        from voxcpm import VoxCPM
        print("Loading VoxCPM2 model...")
        _model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False, optimize=False)
        print("Warming up model...")
        try:
            _model.generate(text="Warming up.", cfg_value=2.0, inference_timesteps=10, max_len=32768)
        except Exception as e:
            print(f"Warm-up generation failed (continuing anyway): {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        print("Model ready.")
    return _model


# ---- settings file ----------------------------------------------------------------
def read_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"engine": "voxcpm", "model_id": MODEL_ID, "profiles": []}


def write_settings(data):
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, SETTINGS_FILE)


def profile_id(voice, lang):
    return f"{voice}__{lang.lower()}"


def get_profile(voice, lang):
    for p in read_settings()["profiles"]:
        if p["id"] == profile_id(voice, lang):
            return p
    return None


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")


def profile_dir(voice, lang):
    return os.path.join(PROFILES_DIR, safe_name(profile_id(voice, lang)))


def preview_file_name(eid, lang):
    """Preview mp3 name from a voice id/name and language, e.g. Stefan_German.mp3."""
    return f"{eid}_{LANG_NAMES.get(lang.upper(), lang.upper())}.mp3"


def preview_filename(voice, lang):
    return preview_file_name(voice_id(voice, read_meta()), lang)


def find_preview(voice, lang):
    """The saved preview of a profile: current name, legacy preview.mp3, or any mp3 in its folder
    (a voice may have been renamed after the preview was made)."""
    d = profile_dir(voice, lang)
    for cand in (os.path.join(d, preview_filename(voice, lang)), os.path.join(d, "preview.mp3")):
        if os.path.exists(cand):
            return cand
    others = sorted(glob.glob(os.path.join(d, "*.mp3")))
    return others[0] if others else None


# ---- synthesis --------------------------------------------------------------------
STYLE_TAG_RE = re.compile(r"^\(([^)]*)\)")


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def with_style(text, style):
    """VoxCPM2 style tags are glued to the first word, no space. Skip if text has its own."""
    text, style = text.strip(), clean_style(style)
    if not style or STYLE_TAG_RE.match(text):
        return text
    return f"({style}){text}"


def synth(text, reference_audio, cfg_value, steps, denoise):
    """Sentence-by-sentence, each continuing from the previous one (same as voxcpm_ui.py).
    The style tag stays on the first sentence only, the chaining carries the tone."""
    import torch
    model = get_model()
    m = STYLE_TAG_RE.match(text.strip())
    style_tag = m.group(0) if m else None
    body = text.strip()[m.end():] if m else text.strip()
    sentences = split_sentences(body)
    if not sentences:
        raise gr.Error("No text to synthesize.")
    if style_tag:
        sentences[0] = f"{style_tag}{sentences[0]}"

    sr = model.tts_model.sample_rate
    gap = np.zeros(int(sr * 0.25), dtype=np.float32)
    chunks, prompt_wav, prompt_text, tmp_files = [], None, None, []
    try:
        for i, sentence in enumerate(sentences):
            try:
                wav = model.generate(
                    text=sentence,
                    reference_wav_path=reference_audio or None,
                    prompt_wav_path=prompt_wav,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=int(steps),
                    denoise=denoise,
                    max_len=32768,
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            chunks.append(wav)
            if i < len(sentences) - 1:
                chunks.append(gap)
            p = tempfile.mktemp(suffix=".wav")
            sf.write(p, wav, sr)
            tmp_files.append(p)
            prompt_wav, prompt_text = p, sentence
        return np.concatenate(chunks), sr
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass


def to_mp3(wav, sr, out_path):
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, wav, sr)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-codec:a", "libmp3lame", "-q:a", "2", out_path],
            check=True,
        )
    except FileNotFoundError:
        raise gr.Error("ffmpeg not found, install it to make mp3 previews.")
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"ffmpeg failed: {e}")
    finally:
        os.remove(tmp)


# ---- UI handlers ------------------------------------------------------------------
def load_profile(voice, lang):
    ref = VOICES.get(voice)
    p = get_profile(voice, lang) if voice else None
    if p:
        note = f"Loaded saved profile {p['id']} (status: {p['status']})."
        if voice in read_flags():
            note += " This voice is marked BAD."
        return (ref, p["cfg_value"], p["inference_timesteps"], p["denoise"], p["style"] or "",
                p["status"], p["notes"], note)
    note = "No saved profile for this combination yet, defaults loaded." if voice else ""
    if voice in read_flags():
        note += " This voice is marked BAD."
    return ref, 2.0, 10, False, "", "untested", "", note


def on_lang(lang, voice, use_sample, text, show_bad):
    voice = voice if voice in voices_for(lang, show_bad) else None
    return (gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice),
            SAMPLES[lang] if use_sample else text)


def on_show_bad(lang, voice, show_bad):
    voice = voice if voice in voices_for(lang, show_bad) else None
    return gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice)


def mark_bad(voice, lang, show_bad):
    if not voice:
        raise gr.Error("Pick a voice first.")
    flags = read_flags()
    flags[voice] = {"marked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_flags(flags)
    note = f"{voice} marked BAD: hidden from the voice list and the API."
    if show_bad:
        note = f"{voice} marked BAD (still shown because 'Show voices marked BAD' is on)."
    return gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice if show_bad else None), note


def unmark_bad(voice, lang, show_bad):
    if not voice:
        raise gr.Error("Pick a voice first (tick 'Show voices marked BAD' to find it).")
    flags = read_flags()
    if voice not in flags:
        raise gr.Error(f"{voice} is not marked BAD.")
    del flags[voice]
    write_flags(flags)
    return gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice), f"{voice} is no longer marked BAD."


def load_meta(voice):
    m = read_meta().get(voice, {}) if voice else {}
    return m.get("name", ""), m.get("gender") or "(not set)"


def apply_name(voice, name, gender):
    """Validate and store the display name + gender of a voice. Returns a status sentence."""
    if not voice:
        raise gr.Error("Pick a voice first.")
    name = (name or "").strip()
    meta = read_meta()
    if name:
        eid = safe_name(name)
        if not eid:
            raise gr.Error("Use letters and numbers in the name.")
        for other in VOICES:
            if other != voice and voice_id(other, {**meta, voice: {"name": name}}) == eid:
                raise gr.Error(f"The name {name} is already used by {other}. Names must be unique because they "
                               "become the voice id in the API.")
    entry = {"name": name, "gender": "" if gender == "(not set)" else gender}
    if entry["name"] or entry["gender"]:
        meta[voice] = {k: v for k, v in entry.items() if v}
    else:
        meta.pop(voice, None)
    write_meta(meta)
    return f"{voice} is now called {name}." if name else f"Cleared the name of {voice}."


def set_name(voice, name, gender, lang, show_bad):
    note = apply_name(voice, name, gender)
    return gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice), note


def apply_avatar(voice, image_path):
    """Store an uploaded avatar unless it is the one already saved. Returns the stored path (or None)."""
    stored = avatar_path(voice)
    if not image_path:
        return stored                      # nothing chosen: keep what is there (use "Remove avatar" to delete)
    if stored and os.path.exists(image_path):
        with open(image_path, "rb") as a, open(stored, "rb") as b:
            if a.read() == b.read():
                return stored              # the component just shows the saved avatar
    dest = os.path.join(AVATARS_DIR, safe_name(voice) + ".png")
    try:
        normalize_avatar(image_path, dest)
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Could not read that image (use PNG, JPG or WebP).")
    return dest


def save_details_only(voice, lang, name, gender, avatar_in, show_bad):
    """Name, gender and avatar of a voice without saving a profile (no preview needed)."""
    note = apply_name(voice, name, gender)
    av = apply_avatar(voice, avatar_in)
    return (gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice),
            f"Saved: {note} Avatar: {'saved' if av else 'none'}.", av)


def save_all(voice, lang, text, ref, cfg, steps, denoise, style, status, notes, name, gender, avatar_in, show_bad):
    """Save profile + the voice's name, gender and avatar together. Checks first, so nothing is half saved."""
    if not voice:
        raise gr.Error("Pick a voice first.")
    if not find_preview(voice, lang):
        raise gr.Error("Generate a preview first, so the saved settings match a preview you have heard.")
    apply_name(voice, name, gender)
    av = apply_avatar(voice, avatar_in)
    msg, settings_path = save_profile(voice, lang, text, ref, cfg, steps, denoise, style, status, notes)
    return (msg, settings_path, gr.Dropdown(choices=voice_choices(lang, show_bad), value=voice), av)


def save_avatar(image_path, voice):
    if not voice:
        raise gr.Error("Pick a voice first, then upload its avatar.")
    if not image_path:
        return None, ""
    dest = os.path.join(AVATARS_DIR, safe_name(voice) + ".png")
    try:
        normalize_avatar(image_path, dest)
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Could not read that image (use PNG, JPG or WebP).")
    return dest, f"Avatar saved for {voice} (square crop, {AVATAR_SIZE}x{AVATAR_SIZE} PNG)."


def load_avatar(voice):
    return avatar_path(voice) if voice else None


def remove_avatar(voice):
    if not voice:
        raise gr.Error("Pick a voice first.")
    if avatar_path(voice):
        os.remove(avatar_path(voice))
        return None, f"Removed the avatar of {voice}."
    return None, f"{voice} has no avatar."


def same_file(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def save_summary(voice, lang, cfg, steps, denoise, style, status, notes, name, gender, avatar_in):
    """Everything 'Save profile' will store, as plain text, so nothing is a surprise."""
    if not voice:
        return "Pick a voice to see what will be saved."
    name_s = (name or "").strip()
    meta = read_meta()
    eid = safe_name(name_s) if name_s else safe_name(voice)
    problems = []
    if name_s:
        if not eid:
            problems.append("The name needs letters or numbers.")
        else:
            for other in VOICES:
                if other != voice and voice_id(other, {**meta, voice: {"name": name_s}}) == eid:
                    problems.append(f"The name {name_s} is already used by {other}.")
                    break
    stored = avatar_path(voice)
    if avatar_in and (not stored or not os.path.exists(avatar_in) or not same_file(avatar_in, stored)):
        avatar = "new image, will be saved as a 512x512 PNG"
    else:
        avatar = "saved (512x512 PNG)" if stored else "none"
    preview = find_preview(voice, lang)
    if not preview:
        problems.append("No preview yet: generate one first, the profile can only be saved after you heard it.")
    style_s = clean_style(style)
    mood = mood_for(style)
    mood_txt = "none" if not style_s else f"{style_s}" + (f"   (preset: {mood})" if mood not in ("(none)", MOOD_CUSTOM) else "   (custom)")
    lines = [
        f"Profile id in the API:  {eid}__{lang.lower()}",
        f"Voice id:               {eid}" + (f"   (technical id: {voice})" if eid != voice else ""),
        f"Language:               {LANG_NAMES.get(lang.upper(), lang)} ({lang.lower()})",
        f"Mood / style:           {mood_txt}",
        f"cfg_value:              {cfg}",
        f"inference_timesteps:    {int(steps)}",
        f"denoise:                {'on' if denoise else 'off'}",
        f"Status:                 {status}",
        f"Notes:                  {(notes or '').strip() or '-'}",
        f"Voice name / gender:    {name_s or '-'} / {'-' if gender in (None, '(not set)') else gender}",
        f"Avatar:                 {avatar}",
        f"Preview file:           {preview_file_name(eid, lang)}" + ("" if preview else "   (not generated yet)"),
    ]
    if problems:
        lines += ["", "To fix before saving:"] + [f"  - {t}" for t in problems]
    return "\n".join(lines)


def deletable_files(voice):
    """Files of a voice we may delete: our own custom voices and the FLEURS candidates.
    Never VCTK, that folder belongs to AllTalk and other tools use it."""
    path = VOICES.get(voice)
    if not path:
        return None
    real = os.path.realpath(path)
    if real.startswith(os.path.realpath(CUSTOM_DIR) + os.sep):
        return [real, os.path.join(CUSTOM_DIR, voice + ".json")]
    if real.startswith(os.path.realpath(CANDIDATES_DIR) + os.sep):
        return [real, os.path.join(CANDIDATES_DIR, voice + ".reference.txt"),
                os.path.join(TESTS_DIR, voice + "__test.wav")]
    return None


def ask_delete(voice):
    if not voice:
        raise gr.Error("Pick a voice first.")
    if deletable_files(voice) is None:
        raise gr.Error(f"{voice} can't be deleted here (VCTK voices live in AllTalk's voices folder and "
                       "other tools use them). Mark it BAD instead.")
    n = sum(1 for p in read_settings()["profiles"] if p["voice"] == voice)
    return (gr.Row(visible=True),
            f"Delete {voice}? This removes its audio file(s) and its {n} saved profile(s) with settings and "
            "previews. It can't be undone. Press 'Yes, delete it permanently' to confirm.",
            voice)


def cancel_delete():
    return gr.Row(visible=False), "Delete cancelled.", None


def confirm_delete(voice, lang, show_bad):
    files = deletable_files(voice) if voice else None
    if files is None:
        raise gr.Error("Nothing to delete. Press 'Delete voice...' first.")
    settings = read_settings()
    removed = [p for p in settings["profiles"] if p["voice"] == voice]
    settings["profiles"] = [p for p in settings["profiles"] if p["voice"] != voice]
    write_settings(settings)
    for l in SAMPLES:  # also previews that were generated but never saved as a profile
        shutil.rmtree(profile_dir(voice, l), ignore_errors=True)
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    flags = read_flags()
    if flags.pop(voice, None) is not None:
        write_flags(flags)
    meta = read_meta()
    if meta.pop(voice, None) is not None:
        write_meta(meta)
    if avatar_path(voice):
        os.remove(avatar_path(voice))
    refresh_voices()
    return (gr.Row(visible=False), gr.Dropdown(choices=voice_choices(lang, show_bad), value=None),
            f"Deleted {voice} and {len(removed)} profile(s).", None)


def on_mood(preset, style):
    return MOODS[preset] if preset in MOODS else style  # '(custom)' keeps what was typed


def on_sample(lang, use_sample, text):
    return SAMPLES[lang] if use_sample else text


def add_voice(audio_path, name, language, source, transcript, cur_lang, show_bad=False):
    """Turn an uploaded/recorded clip into a named voice in CUSTOM_DIR (wav + provenance json)."""
    if not audio_path:
        raise gr.Error("Upload or record a reference clip first.")
    name = safe_name(name or "")
    if not name:
        raise gr.Error("Give the voice a name (letters, numbers, - and _).")
    if name in VOICES:
        raise gr.Error(f"A voice called {name} already exists, pick another name.")
    source = (source or "").strip()
    if len(source) < 5:
        raise gr.Error("Say where the recording comes from (own voice, consent given by whom, license). "
                       "This is required so every voice can be traced.")

    dest = os.path.join(CUSTOM_DIR, name + ".wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path, "-ac", "1",
                        "-c:a", "pcm_s16le", dest], check=True)
    except FileNotFoundError:
        raise gr.Error("ffmpeg not found, install it to add voices.")
    except subprocess.CalledProcessError:
        raise gr.Error("Could not read that audio file.")
    dur = sf.info(dest).duration
    if dur < 3 or dur > 60:
        os.remove(dest)
        raise gr.Error(f"The clip is {dur:.1f} s long. It needs to be between 3 and 60 s (6 to 30 s works best).")

    meta = {"name": name, "language": language, "source": source,
            "transcript": (transcript or "").strip() or None, "duration_s": round(dur, 1),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(CUSTOM_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    refresh_voices()

    lang = language.upper() if language.upper() in SAMPLES else cur_lang
    note = f"Added voice {name} ({dur:.1f} s) in {CUSTOM_DIR} and selected it below."
    if not 6 <= dur <= 30:
        note += " Note: 6 to 30 s of clean speech works best for cloning."
    return (gr.Dropdown(choices=list(SAMPLES), value=lang),
            gr.Dropdown(choices=voice_choices(lang, show_bad), value=name), note)


def generate_preview(voice, lang, text, ref, cfg, steps, denoise, style, progress=gr.Progress()):
    if not voice:
        raise gr.Error("Pick a voice first.")
    progress(0, desc="Generating preview (the first one also loads the model)...")
    wav, sr = synth(with_style(text, style), ref, cfg, steps, denoise)
    d = profile_dir(voice, lang)
    os.makedirs(d, exist_ok=True)
    mp3_path = os.path.join(d, preview_filename(voice, lang))
    to_mp3(wav, sr, mp3_path)
    for old in glob.glob(os.path.join(d, "*.mp3")):  # keep one preview per profile (maybe under an older name)
        if old != mp3_path:
            os.remove(old)
    return mp3_path, f"Preview saved: {mp3_path}"


def save_profile(voice, lang, text, ref, cfg, steps, denoise, style, status, notes):
    if not voice:
        raise gr.Error("Pick a voice first.")
    d = profile_dir(voice, lang)
    preview = find_preview(voice, lang)
    if not preview:
        raise gr.Error("Generate a preview first, so the saved settings match a preview you have heard.")
    wanted = os.path.join(d, preview_filename(voice, lang))   # e.g. Stefan_German.mp3 (name may be new)
    if preview != wanted:
        os.replace(preview, wanted)
        preview = wanted
    ref_file = None
    if ref:
        ref_file = "reference" + os.path.splitext(ref)[1]
        shutil.copyfile(ref, os.path.join(d, ref_file))

    profile = {
        "id": profile_id(voice, lang),
        "voice": voice,
        "language": lang.lower(),
        "cfg_value": cfg,
        "inference_timesteps": int(steps),
        "denoise": denoise,
        "style": clean_style(style) or None,
        "status": status,
        "notes": notes,
    }
    with open(os.path.join(d, "profile.json"), "w", encoding="utf-8") as f:
        json.dump({**profile, "engine": "voxcpm", "model_id": MODEL_ID, "preview_text": text,
                   "preview_audio": os.path.basename(preview), "reference_audio": ref_file,
                   "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, ensure_ascii=False, indent=2)

    settings = read_settings()
    settings["profiles"] = [p for p in settings["profiles"] if p["id"] != profile["id"]] + [profile]
    settings["profiles"].sort(key=lambda p: (p["voice"], p["language"]))
    write_settings(settings)
    return (f"Saved {profile['id']}: {len(settings['profiles'])} profile(s) in {SETTINGS_FILE}",
            SETTINGS_FILE)


# ---- API for other programs (AIcove, scripts) ---------------------------------------
# Two calls besides the page: voices() lists the saved profiles, speak(text, profile) says a
# text with one of them (its voice, settings and mood) and returns a WAV. Plain HTTP, see
# README.md. Nothing on the page changes.
def api_voices() -> list:
    """The saved profiles: id, display name, language, gender, mood, status."""
    meta = read_meta()
    out = []
    for p in read_settings()["profiles"]:
        m = meta.get(p["voice"], {})
        out.append({"id": p["id"], "name": m.get("name") or p["voice"], "language": p["language"],
                    "gender": m.get("gender") or "", "style": p.get("style") or "",
                    "status": p.get("status") or "", "bad": p["voice"] in read_flags()})
    return out


def profile_reference(p):
    """The reference clip of a saved profile: its own copy, else the voice's clip."""
    saved = os.path.join(PROFILES_DIR, safe_name(p["id"]), "profile.json")
    try:
        with open(saved, encoding="utf-8") as f:
            ref = json.load(f).get("reference_audio")
        if ref and os.path.exists(os.path.join(os.path.dirname(saved), ref)):
            return os.path.join(os.path.dirname(saved), ref)
    except (OSError, json.JSONDecodeError):
        pass
    refresh_voices()
    return VOICES.get(p["voice"])


def api_speak(text, profile):
    """Say text with a saved profile; returns the path of a WAV file."""
    p = next((x for x in read_settings()["profiles"] if x["id"] == profile), None)
    if p is None:
        raise gr.Error(f"No saved profile {profile!r}. See voices().")
    if not (text or "").strip():
        raise gr.Error("No text to say.")
    wav, sr = synth(with_style(text, p.get("style") or ""), profile_reference(p),
                    p["cfg_value"], p["inference_timesteps"], p["denoise"])
    fd, out = tempfile.mkstemp(suffix=".wav", prefix="speak-")
    os.close(fd)
    sf.write(out, wav, sr)
    return out


with gr.Blocks(title="voxcpm-voices") as demo:
    gr.Markdown(
        "# voxcpm-voices\n"
        "Pick language and voice, tune the settings, generate a preview, save the profile. "
        f"Profiles go into one settings file, `{SETTINGS_FILE}`, "
        f"each profile's `preview.mp3` and reference clip are saved in `{PROFILES_DIR}`."
    )
    with gr.Accordion("Add a new voice from your own recording", open=False):
        new_audio = gr.Audio(label="Reference clip (6 to 30 s of clean speech, one speaker)", type="filepath",
                             sources=["upload", "microphone"])
        with gr.Row():
            new_name = gr.Textbox(label="Voice name (e.g. anna_de)")
            new_lang = gr.Dropdown(choices=["any"] + list(SAMPLES), value="any",
                                   label="Language of the recording (any = neutral; every voice works for every language)")
        new_source = gr.Textbox(
            label="Source / consent / license (required)",
            placeholder="e.g. own voice | Anna, written consent for synthetic use, 2026-09 | CC BY 4.0, link")
        new_transcript = gr.Textbox(label="What is said in the clip (optional)")
        add_btn = gr.Button("Add voice")
        add_msg = gr.Textbox(label="Result", interactive=False)
    with gr.Row():
        lang_dd = gr.Dropdown(choices=list(SAMPLES), value="EN", label="Language")
        voice_dd = gr.Dropdown(choices=voice_choices("EN"), value=None, label="Voice")
    with gr.Row():
        show_bad_cb = gr.Checkbox(label="Show voices marked BAD", value=False)
        bad_btn = gr.Button("Mark voice as BAD")
        unbad_btn = gr.Button("Unmark BAD")
        del_btn = gr.Button("Delete voice...", variant="stop")
    with gr.Row(visible=False) as confirm_row:
        confirm_btn = gr.Button("Yes, delete it permanently", variant="stop")
        cancel_btn = gr.Button("Cancel")
    voice_msg = gr.Markdown("")  # plain text line for the BAD / delete messages, empty until there is one
    pending = gr.State(None)
    ref_audio = gr.Audio(label="Reference voice (set by the voice choice)", type="filepath", interactive=False)
    use_sample = gr.Checkbox(label="Use the sample sentence for this language", value=True)
    text_in = gr.Textbox(label="Preview text", lines=3, value=SAMPLES["EN"])
    with gr.Row():
        cfg_slider = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="cfg_value")
        steps_slider = gr.Slider(4, 30, value=10, step=1, label="inference_timesteps")
    with gr.Row():
        mood_dd = gr.Dropdown(choices=list(MOODS) + [MOOD_CUSTOM], value="(none)",
                              label="Mood preset (fills the field on the right, you can edit it)")
        style_tb = gr.Textbox(label="Mood / style words (applied automatically, no brackets needed)",
                              placeholder="e.g. calm, slightly slower",
                              info="Describes how the voice should sound, for example friendly or calm. "
                                   "Pick a preset or type your own words. Results vary by voice and "
                                   "language, so listen to the preview.")
    denoise_check = gr.Checkbox(label="Denoise reference audio", value=False)
    preview_btn = gr.Button("Generate preview (saves <voice name>_<Language>.mp3)", variant="primary")
    audio_out = gr.Audio(label="Preview", type="filepath")
    with gr.Row():
        status_dd = gr.Dropdown(choices=STATUSES, value="untested", label="Status")
        notes_tb = gr.Textbox(label="Notes")
    gr.Markdown("**Voice details** (they belong to the voice and apply in every language; saved with the profile)")
    with gr.Row():
        name_tb = gr.Textbox(label="Voice name (shown in lists, and used as the voice id in the API)",
                             placeholder="e.g. Stefan")
        gender_dd = gr.Dropdown(choices=["(not set)", "male", "female"], value="(not set)", label="Gender")
    with gr.Row():
        avatar_img = gr.Image(label="Avatar (any image; saved as a square 512x512 PNG)",
                              type="filepath", sources=["upload"], height=180)
        with gr.Column():
            remove_avatar_btn = gr.Button("Remove avatar")
            name_btn = gr.Button("Save voice details only (no preview needed)")
    summary_tb = gr.Textbox(label="What will be saved (updates as you change things)", lines=13,
                            interactive=False, value="Pick a voice to see what will be saved.")
    save_btn = gr.Button("Save profile (also saves name, gender and avatar)", variant="primary")
    msg = gr.Textbox(label="Result", interactive=False)
    settings_file = gr.File(label="Settings file (voice_profiles.json)", value=SETTINGS_FILE if os.path.exists(SETTINGS_FILE) else None,
                            interactive=False)

    profile_outputs = [ref_audio, cfg_slider, steps_slider, denoise_check, style_tb, status_dd, notes_tb, msg]
    lang_dd.change(on_lang, [lang_dd, voice_dd, use_sample, text_in, show_bad_cb], [voice_dd, text_in]) \
        .then(load_profile, [voice_dd, lang_dd], profile_outputs)
    voice_dd.change(load_profile, [voice_dd, lang_dd], profile_outputs)
    voice_dd.change(lambda: (gr.Row(visible=False), None), None, [confirm_row, pending])  # stale confirm
    show_bad_cb.change(on_show_bad, [lang_dd, voice_dd, show_bad_cb], voice_dd)
    voice_dd.change(load_meta, voice_dd, [name_tb, gender_dd])
    name_btn.click(save_details_only, [voice_dd, lang_dd, name_tb, gender_dd, avatar_img, show_bad_cb],
                   [voice_dd, msg, avatar_img])
    voice_dd.change(load_avatar, voice_dd, avatar_img)
    remove_avatar_btn.click(remove_avatar, voice_dd, [avatar_img, msg])
    mood_dd.input(on_mood, [mood_dd, style_tb], style_tb)
    style_tb.change(mood_for, style_tb, mood_dd)
    bad_btn.click(mark_bad, [voice_dd, lang_dd, show_bad_cb], [voice_dd, voice_msg])
    unbad_btn.click(unmark_bad, [voice_dd, lang_dd, show_bad_cb], [voice_dd, voice_msg])
    del_btn.click(ask_delete, voice_dd, [confirm_row, voice_msg, pending])
    cancel_btn.click(cancel_delete, None, [confirm_row, voice_msg, pending])
    confirm_btn.click(confirm_delete, [pending, lang_dd, show_bad_cb], [confirm_row, voice_dd, voice_msg, pending])
    use_sample.change(on_sample, [lang_dd, use_sample, text_in], text_in)
    add_btn.click(add_voice, [new_audio, new_name, new_lang, new_source, new_transcript, lang_dd, show_bad_cb],
                  [lang_dd, voice_dd, add_msg])
    preview_btn.click(generate_preview,
                      [voice_dd, lang_dd, text_in, ref_audio, cfg_slider, steps_slider, denoise_check, style_tb],
                      [audio_out, msg], concurrency_id="voxcpm")
    save_btn.click(save_all,
                   [voice_dd, lang_dd, text_in, ref_audio, cfg_slider, steps_slider, denoise_check, style_tb,
                    status_dd, notes_tb, name_tb, gender_dd, avatar_img, show_bad_cb],
                   [msg, settings_file, voice_dd, avatar_img])

    summary_inputs = [voice_dd, lang_dd, cfg_slider, steps_slider, denoise_check, style_tb, status_dd, notes_tb,
                      name_tb, gender_dd, avatar_img]
    for _component in summary_inputs:
        _component.change(save_summary, summary_inputs, summary_tb)

    # The API (no visible parts). One model: speak and the page's preview take turns.
    with gr.Group(visible=False):
        api_text, api_profile = gr.Textbox(), gr.Textbox()
        api_audio = gr.Audio(type="filepath")
        api_list = gr.JSON()
        api_speak_btn, api_list_btn = gr.Button(), gr.Button()
    api_speak_btn.click(api_speak, [api_text, api_profile], api_audio, api_name="speak", concurrency_id="voxcpm")
    api_list_btn.click(api_voices, None, api_list, api_name="voices")

demo.launch(server_name="127.0.0.1", server_port=PORT)
