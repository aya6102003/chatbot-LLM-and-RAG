import subprocess
from pathlib import Path

# text file containing URLs (one URL per line)
URLS_FILE = "urls.txt"

# optional extra crawler arguments
EXTRA_ARGS = [
    "--full"
]

def load_urls(file_path):
    urls = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            urls.append(line)

    return urls


def main():
    urls = load_urls(URLS_FILE)

    if not urls:
        print("No URLs found.")
        return

    print(f"Found {len(urls)} URLs\n")

    for i, url in enumerate(urls, 1):
        print("=" * 80)
        print(f"[{i}/{len(urls)}] Crawling: {url}")
        print("=" * 80)

        cmd = [
            "python",
            "crawler.py",
            "--url",
            url,
            *EXTRA_ARGS
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Finished: {url}\n")

        except subprocess.CalledProcessError as e:
            print(f"Failed: {url}")
            print(e)


if __name__ == "__main__":
    main()