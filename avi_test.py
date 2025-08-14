import json
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("configs.json", "r") as f:
        return json.load(f)

def get_access_token(config):
    vi_config = config['video_indexer']
    token_url = f"https://api.videoindexer.ai/Auth/{vi_config['location']}/Accounts/{vi_config['account_id']}/AccessTokenWithPermission"
    headers = {"Ocp-Apim-Subscription-Key": vi_config['subscription_key']}
    params = {"permission": "Owner"}

    logger.info("Requesting access token...")
    resp = requests.get(token_url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.text.strip('"')

def upload_video(config, access_token, video_path, name, description):
    vi_config = config['video_indexer']
    upload_url = f"https://api.videoindexer.ai/{vi_config['location']}/Accounts/{vi_config['account_id']}/Videos"
    params = {
        "accessToken": access_token,
        "name": name,
        "description": description,
        "privacy": config['processing']['privacy'],
        "indexingPreset": "Advanced",
        "streamingPreset": "Default",
        "detectSourceLanguage": "true",
        "multiLanguage": "true"
    }

    with open(video_path, "rb") as f:
        files = {"file": (Path(video_path).name, f, "video/mp4")}
        resp = requests.post(upload_url, params=params, files=files)
    resp.raise_for_status()
    return resp.json()["id"]

def wait_for_indexing(config, access_token, video_id):
    vi_config = config['video_indexer']
    url = f"https://api.videoindexer.ai/{vi_config['location']}/Accounts/{vi_config['account_id']}/Videos/{video_id}/Index"
    params = {"accessToken": access_token, "includeSummarizedInsights": "true"}
    timeout = config['processing']['timeout_seconds']
    start_time = time.time()

    logger.info("Waiting for indexing to finish...")
    while True:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "").lower()

        if state == "processed":
            logger.info("Indexing complete.")
            return data
        elif state == "failed":
            raise RuntimeError("Indexing failed.")

        if time.time() - start_time > timeout:
            raise TimeoutError("Indexing timed out.")

        time.sleep(10)

if __name__ == "__main__":
    config = load_config()
    video_path = "/Users/mohammedkhambhati/Desktop/VIDA/test.mp4"  # CHANGE THIS
    video_name = "Test Video"
    video_description = "AVI test indexing"

    access_token = get_access_token(config)
    video_id = upload_video(config, access_token, video_path, video_name, video_description)
    data = wait_for_indexing(config, access_token, video_id)

    with open("avi_index_output.txt", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved AVI index data to avi_index_output.txt")
