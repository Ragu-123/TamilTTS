"""
TamilTTS Dataset Preprocessing & Forced Alignment CLI
======================================================
Usage:
  # Quick check on 2 samples from each dataset:
  python preprocess.py --limit_samples 2

  # Full run on all 75k+ samples:
  python preprocess.py
"""
import os, sys, io, glob, time, shutil, argparse, urllib.request, torch
import soundfile as sf
import pyarrow.parquet as pq

from preprocess.g2g import segment_tamil_g2g, load_g2g_dictionary
from preprocess.align import run_mfa_alignment, extract_durations_from_textgrid

DICT_URL = "https://github.com/AI4Bharat/IndicMFA/releases/download/Tamil/Tamil_Dictionary_g2g.txt"
MODEL_URL = "https://github.com/AI4Bharat/IndicMFA/releases/download/Tamil/Tamil_Acoustic_Model.zip"

def download_indic_mfa_assets(dest_dir="indic_mfa_tamil"):
    os.makedirs(dest_dir, exist_ok=True)
    dict_path = os.path.join(dest_dir, "Tamil_Dictionary_g2g.txt")
    model_zip = os.path.join(dest_dir, "Tamil_Acoustic_Model.zip")

    if not os.path.exists(dict_path):
        print(f"Downloading Tamil G2G Dictionary from {DICT_URL}...")
        urllib.request.urlretrieve(DICT_URL, dict_path)
        print("  ✓ Dictionary ready.")

    if not os.path.exists(model_zip):
        print(f"Downloading Tamil Acoustic Model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, model_zip)
        print("  ✓ Acoustic Model ready.")

    return dict_path, model_zip

def main():
    parser = argparse.ArgumentParser(description="TamilTTS IndicMFA Preprocessor")
    parser.add_argument("--dataset_dirs", nargs="+", default=[
        "/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil",
        "/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil",
    ])
    parser.add_argument("--mfa_bin", default="/kaggle/working/mfa_env/bin/mfa")
    parser.add_argument("--assets_dir", default="/kaggle/working/indic_mfa_tamil")
    parser.add_argument("--output_file", default="/kaggle/working/tamil_mfa_durations_76k.pt")
    parser.add_argument("--temp_corpus", default="/kaggle/working/temp_mfa_corpus")
    parser.add_argument("--temp_textgrids", default="/kaggle/working/temp_mfa_textgrids")
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--limit_samples", type=int, default=0, help="0 for all, or N for quick check")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    dict_path, model_path = download_indic_mfa_assets(args.assets_dir)
    g2g_entries = load_g2g_dictionary(dict_path)
    print(f"Loaded {len(g2g_entries)} graphemes in IndicMFA G2G dictionary.")

    print("\n" + "=" * 60)
    print("  TamilTTS Industry-Standard Preprocessing & Alignment")
    print("=" * 60)

    # 1. Find Parquet files
    parquet_files = []
    for d in args.dataset_dirs:
        found = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        if found:
            print(f"  ✓ Found {len(found)} parquet files in '{d}'")
            parquet_files.extend(found)
        else:
            print(f"  ⚠️ No parquet files found in '{d}'")

    if not parquet_files:
        print("No parquet files found! Check dataset paths.")
        return

    # 2. Index samples (support limit_samples per dataset or global)
    all_samples = []
    print("Indexing parquet rows...")
    for pf_path in parquet_files:
        try:
            pf = pq.ParquetFile(pf_path)
            for rg_idx in range(pf.num_row_groups):
                num_rows = pf.metadata.row_group(rg_idx).num_rows
                for r in range(num_rows):
                    all_samples.append((pf_path, rg_idx, r))
                    if args.limit_samples > 0 and len(all_samples) >= args.limit_samples:
                        break
                if args.limit_samples > 0 and len(all_samples) >= args.limit_samples:
                    break
            if args.limit_samples > 0 and len(all_samples) >= args.limit_samples:
                break
        except Exception as e:
            print(f"Could not index {pf_path}: {e}")

    total_samples = len(all_samples)
    print(f"Total samples to process: {total_samples:,}")

    # 3. Load checkpoint if exists
    durations_cache = {}
    if os.path.exists(args.output_file):
        try:
            durations_cache = torch.load(args.output_file)
            print(f"Resuming: {len(durations_cache):,} samples already aligned in checkpoint.")
        except Exception:
            pass

    chunk_size = args.chunk_size if args.limit_samples == 0 else min(args.chunk_size, args.limit_samples)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_i = chunk_idx * chunk_size
        end_i = min(start_i + chunk_size, total_samples)
        chunk_items = all_samples[start_i:end_i]

        to_process = []
        for item in chunk_items:
            key = f"{os.path.basename(item[0])}_rg{item[1]}_r{item[2]}"
            if key not in durations_cache:
                to_process.append((key, item[0], item[1], item[2]))

        if not to_process:
            continue

        print(f"\n--- Chunk {chunk_idx+1}/{num_chunks} ({len(to_process)} samples) ---")
        shutil.rmtree(args.temp_corpus, ignore_errors=True)
        shutil.rmtree(args.temp_textgrids, ignore_errors=True)
        os.makedirs(args.temp_corpus, exist_ok=True)
        os.makedirs(args.temp_textgrids, exist_ok=True)

        exported = 0
        cur_pf, cur_table = None, None
        for key, p_path, rg_idx, r_idx in to_process:
            try:
                if cur_pf != (p_path, rg_idx):
                    pf = pq.ParquetFile(p_path)
                    cur_table = pf.read_row_group(rg_idx)
                    cur_pf = (p_path, rg_idx)

                row = {col: cur_table[col][r_idx].as_py() for col in cur_table.column_names}
                text = (row.get("normalized") or row.get("text") or row.get("verbatim") or "").strip()
                if not text: continue

                audio_data = row.get("audio")
                raw_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None
                if not raw_bytes or len(raw_bytes) < 100: continue

                arr, orig_sr = sf.read(io.BytesIO(raw_bytes))
                if arr.ndim > 1: arr = arr.mean(axis=1)

                wav_p = os.path.join(args.temp_corpus, f"{key}.wav")
                txt_p = os.path.join(args.temp_corpus, f"{key}.txt")

                sf.write(wav_p, arr, orig_sr)
                g2g_str = segment_tamil_g2g(text, g2g_entries)
                with open(txt_p, "w", encoding="utf-8") as f:
                    f.write(g2g_str)
                exported += 1
            except Exception:
                continue

        if exported == 0:
            continue

        print(f"  Exported {exported} audio/text files. Running MFA aligner...")
        success, elapsed = run_mfa_alignment(
            args.temp_corpus, dict_path, model_path, args.temp_textgrids,
            mfa_bin=args.mfa_bin, jobs=args.jobs
        )

        tg_files = glob.glob(os.path.join(args.temp_textgrids, "*.TextGrid"))
        print(f"  MFA finished in {elapsed:.1f}s. Extracted {len(tg_files)}/{exported} TextGrids.")

        for tg_p in tg_files:
            try:
                key = os.path.splitext(os.path.basename(tg_p))[0]
                durs, toks = extract_durations_from_textgrid(
                    tg_p, hop_length=args.hop_length, sample_rate=args.sample_rate
                )
                if durs:
                    durations_cache[key] = {
                        "durations": durs,
                        "tokens": toks
                    }
                    if args.limit_samples > 0:
                        print(f"  Sample '{key}': {len(durs)} characters aligned, total {sum(durs)} mel frames ({sum(durs)*256/22050:.2f}s)")
                        print(f"    Graphemes: {toks[:10]}...")
                        print(f"    Durations: {durs[:10]}...")
            except Exception:
                continue

        torch.save(durations_cache, args.output_file)
        f_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
        print(f"  Saved Checkpoint: {len(durations_cache):,} total aligned utterances ({f_size_mb:.1f} MB on disk)")

        shutil.rmtree(args.temp_corpus, ignore_errors=True)
        shutil.rmtree(args.temp_textgrids, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  🎉 Preprocessing & Alignment Complete!")
    print(f"  Total aligned samples: {len(durations_cache):,}")
    print(f"  Output saved to: {args.output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
