"""
IndicMFA Alignment & Duration Extraction Module
"""
import os, glob, time, shutil, subprocess, torch
from concurrent.futures import ThreadPoolExecutor

def run_mfa_alignment(corpus_dir, dict_path, model_path, output_dir, mfa_bin="mfa", jobs=4):
    os.makedirs(output_dir, exist_ok=True)
    mfa_dir = os.path.dirname(mfa_bin) if os.path.isabs(mfa_bin) else ""
    mfa_root = os.path.dirname(mfa_dir) if mfa_dir else "/kaggle/working/mfa_env"

    # Auto-detect the actual python binary — exclude -config, .py, .m scripts
    python_bin = "python"
    if mfa_dir:
        import re
        for name in sorted(os.listdir(mfa_dir), reverse=True):
            full = os.path.join(mfa_dir, name)
            if re.match(r"^python3(\.\d+)?$", name) and os.path.isfile(full):
                python_bin = full
                break

    # Set OMP_NUM_THREADS=1 so all 48 workers run at 100% CPU without thread contention
    bash_cmd = (
        f"export CONDA_PREFIX='{mfa_root}'; "
        f"export PATH='{mfa_root}/bin:$PATH'; "
        f"export LD_LIBRARY_PATH='{mfa_root}/lib:$LD_LIBRARY_PATH'; "
        f"export OMP_NUM_THREADS=1; "
        f"export OPENBLAS_NUM_THREADS=1; "
        f"export MKL_NUM_THREADS=1; "
        f"'{python_bin}' -m montreal_forced_aligner.command_line.mfa align "
        f"--clean --use_mp -j {jobs} --single_speaker --no_textgrid_cleanup "
        f"'{corpus_dir}' '{dict_path}' '{model_path}' '{output_dir}'"
    )

    t0 = time.time()
    res = subprocess.run(["bash", "-c", bash_cmd], capture_output=True, text=True)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"\n  ⚠️ MFA Exit Code {res.returncode}:")
        if res.stderr:
            print("  [MFA STDERR]:", res.stderr[-600:])
        if res.stdout:
            print("  [MFA STDOUT]:", res.stdout[-600:])
    return res.returncode == 0, elapsed

def parse_single_textgrid(tg_p, hop_length=256, sample_rate=22050):
    try:
        import textgrid
        tg = textgrid.TextGrid.fromFile(tg_p)
        key = os.path.splitext(os.path.basename(tg_p))[0]
        phones_tier = [t for t in tg.tiers if t.name.lower() in ["phones", "phone", "phonemes", "phoneme", "graphemes", "grapheme"]]
        if not phones_tier and len(tg.tiers) > 0:
            phones_tier = [tg.tiers[-1]]
        if not phones_tier:
            return None, None, None

        frame_dur_sec = hop_length / sample_rate
        durs = []
        toks = []
        for interval in phones_tier[0].intervals:
            mark = interval.mark.strip()
            if mark and mark != "<eps>":
                n_frames = max(1, int(round((interval.maxTime - interval.minTime) / frame_dur_sec)))
                durs.append(n_frames)
                toks.append(mark)
        return key, durs, toks
    except Exception:
        return None, None, None
