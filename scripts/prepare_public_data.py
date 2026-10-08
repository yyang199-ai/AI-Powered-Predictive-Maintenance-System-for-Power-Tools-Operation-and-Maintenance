"""Download and prepare real CWRU records. Run from the repository root."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictive.public_data import prepare_cwru


def main():
    parser = argparse.ArgumentParser(description="CWRU公开振动：下载、抗混叠重采样、小波降噪、FFT特征、文件隔离标注")
    parser.add_argument("--raw-dir", default="data/public/CWRU/raw")
    parser.add_argument("--output-dir", default="artifacts/bearing/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows-per-record", type=int, default=256)
    parser.add_argument("--offline", action="store_true", help="仅使用已经下载的40个原始MAT")
    args = parser.parse_args()
    result = prepare_cwru(args.raw_dir, args.output_dir, args.max_windows_per_record, args.seed, not args.offline)
    print(json.dumps({key: result[key] for key in ("total_windows", "total_records", "record_split", "window_split", "dataset_bytes", "dataset_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
