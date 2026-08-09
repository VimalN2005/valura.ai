import os
import urllib.request
import json
import ssl

def download_file(url, filepath, token):
    print(f"Downloading {url} -> {filepath}...")
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "antigravity-downloader"
        }
    )
    # Ignore SSL certification checks in case of custom proxies/certs issues
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            data = response.read()
            # Try to format json nicely if possible
            try:
                parsed = json.loads(data)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, indent=2)
            except Exception:
                with open(filepath, 'wb') as f:
                    f.write(data)
            print(f"Successfully downloaded to {filepath}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    api_key = "vlr_9MgCCpiMEKBico7RT_N2aqRP-Du-cYOO"
    base_url = "https://ai-arena.twocc.in"
    mode = "practice"

    os.makedirs("data", exist_ok=True)

    download_file(f"{base_url}/v1/rules?mode={mode}", "data/rules.json", api_key)
    download_file(f"{base_url}/v1/book?mode={mode}", "data/book.json", api_key)
    download_file(f"{base_url}/v1/market?mode={mode}", "data/market.json", api_key)

    # Let's inspect the files
    print("\n--- Summary of downloaded files ---")
    for filename in ["rules.json", "book.json", "market.json"]:
        path = os.path.join("data", filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"{filename}: Keys={list(data.keys())}")
                        if "meta" in data:
                            print(f"  meta={data['meta']}")
                    elif isinstance(data, list):
                        print(f"{filename}: List length={len(data)}")
                        if len(data) > 0:
                            print(f"  First item type={type(data[0])}")
                except Exception as e:
                    print(f"Could not parse {filename} as JSON: {e}")
        else:
            print(f"{filename} does not exist.")

if __name__ == "__main__":
    main()
