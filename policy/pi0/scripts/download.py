
# download_pi0_fast_params.py

from pathlib import Path
from openpi.shared import download

REMOTE_PATH = "gs://openpi-assets/checkpoints/pi0_fast_base/params"

def main():
    local_path = download.maybe_download(REMOTE_PATH)
    print(f"Downloaded/cached to: {local_path}")

    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Download reported success, but path does not exist: {p}")

    print("\nTop-level contents:")
    for child in sorted(p.iterdir()):
        print(" -", child)

if __name__ == "__main__":
    main()