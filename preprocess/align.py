"""
IndicMFA Alignment & Duration Extraction Module
"""
import os, glob, time, shutil, subprocess, textgrid, torch

def run_mfa_alignment(corpus_dir, dict_path, model_path, output_dir, mfa_bin="mfa", jobs=4):
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    mfa_dir = os.path.dirname(mfa_bin) if os.path.isabs(mfa_bin) else ""
    if mfa_dir:
        mfa_root = os.path.dirname(mfa_dir)
        env["PATH"] = f"{mfa_dir}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{mfa_root}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        env["CONDA_PREFIX"] = mfa_root

    # Use mfa_env python binary if present to avoid shebang path issues
    python_bin = os.path.join(mfa_dir, "python") if mfa_dir and os.path.exists(os.path.join(mfa_dir, "python")) else (mfa_bin if os.path.exists(mfa_bin) else "mfa")

    if python_bin.endswith("python") or python_bin.endswith("python3"):
        cmd = [
            python_bin, "-m", "montreal_forced_aligner.command_line.mfa", "align",
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
    else:
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
    if res.returncode != 0:
        print(f"\n  ⚠️ MFA Exit Code {res.returncode}:")
        if res.stderr:
            print("  [MFA STDERR]:", res.stderr[-600:])
        if res.stdout:
            print("  [MFA STDOUT]:", res.stdout[-600:])
    return res.returncode == 0, elapsed

def extract_durations_from_textgrid(tg_path, hop_length=256, sample_rate=22050):
    tg = textgrid.TextGrid.fromFile(tg_path)
    phones_tier = [t for t in tg.tiers if t.name.lower() in ["phones", "phone", "phonemes", "phoneme", "graphemes", "grapheme"]]
    if not phones_tier:
        # Fallback to last tier if names differ
        phones_tier = [tg.tiers[-1]] if len(tg.tiers) > 0 else []
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
