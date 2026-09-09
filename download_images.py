import requests
import pandas as pd

import config


def main():
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.CSV_URL)

    success, skipped, failed = 0, 0, 0
    for i, row in df.iterrows():
        dest = config.IMAGE_DIR / f"{row['md5hash']}.jpg"
        if dest.exists():
            skipped += 1
            continue
        try:
            response = requests.get(row["url"], timeout=10)
            response.raise_for_status()
            dest.write_bytes(response.content)
            success += 1
        except Exception:
            failed += 1

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(df)} — {success} downloaded, {failed} failed, {skipped} already cached")

    print(f"\nDone. {success} downloaded, {failed} failed (dead links), {skipped} already cached.")
    print(f"Images saved to {config.IMAGE_DIR}")


if __name__ == "__main__":
    main()