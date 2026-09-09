import requests
import pandas as pd

import config

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def main():
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.CSV_URL)

    success, skipped, failed = 0, 0, 0
    failure_samples = []

    for i, row in df.iterrows():
        dest = config.IMAGE_DIR / f"{row['md5hash']}.jpg"
        if dest.exists():
            skipped += 1
            continue
        try:
            response = requests.get(row["url"], headers=HEADERS, timeout=10, verify=False)
            content_type = response.headers.get("Content-Type", "")
            if response.status_code != 200 or "image" not in content_type:
                raise ValueError(f"status={response.status_code}, content-type={content_type}")
            dest.write_bytes(response.content)
            success += 1
        except Exception as e:
            failed += 1
            if len(failure_samples) < 5:
                failure_samples.append(f"{row['url']} -> {e}")

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(df)} — {success} downloaded, {failed} failed, {skipped} already cached")

    print(f"\nDone. {success} downloaded, {failed} failed (dead links), {skipped} already cached.")
    print(f"Images saved to {config.IMAGE_DIR}")
    if failure_samples:
        print("\nSample failures (first 5):")
        for s in failure_samples:
            print(f"  {s}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()