"""
IndicMFA Alignment & Duration Extraction Module
"""
import os, glob, time, shutil, subprocess, textgrid, torch

def run_mfa_alignment(corpus_dir, dict_path, model_path, output_dir, mfa_bin="mfa", jobs=4):
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    mfa_dir = os.path.dirname(mfa_bin)
    if mfa_dir:
        env["PATH"] = f"{mfa_dir}:{env.get('PATH', '')}"

    cmd = [
        mfa_bin, "align",
        "--clean",
        "--use_mp",
        "-j", str(jobs),
        "--single_speaker",
        "--no_textgrid_cleanup",
        corpus_dir,
        dict_path,
        model_path,
        output_dir
    ]
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    return res.returncode == 0, elapsed

def extract_durations_from_textgrid(tg_path, hop_length=256, sample_rate=22050):
    tg = textgrid.TextGrid.fromFile(tg_path)
    phones_tier = [t for t in tg.tiers if t.name == "phones"]
    if not phones_tier:
        return None, None

    frame_dur_sec = hop_length / sample_rate
    durations = []
    tokens = []
    for interval in phones_tier[0].intervals:
        mark = interval.mark.strip()
        if mark and mark != "<eps>":
            n_frames = max(1, int(round((interval.maxTime - interval.minTime) / frame_dur_sec)))
            durations.append(n_frames)
            tokens.append(mark)
    return durations, tokens
