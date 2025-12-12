import os
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm


# --------------- CONFIG ---------------

# ⚠️ DO NOT hardcode your key in the script if you ever push it to Git.
# Set it as an environment variable instead:
#export YOUTUBE_API_KEY=""
API_KEY = os.environ.get("YOUTUBE_API_KEY", "USE YOUR KEY.")

if not API_KEY or API_KEY == "PUT_YOUR_KEY_HERE":
    raise RuntimeError("Set YOUTUBE_API_KEY env var or put your API key in API_KEY.")

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# How far back to look for videos
DAYS_BACK = 7

# Output CSV
OUTPUT_CSV = "videos_raw.csv"

# Food-related search queries
FOOD_QUERIES = [
    "easy recipe",
    "quick dinner",
    "street food",
    "high protein meal",
    "cheap meals",
    "hostel food",
    "dessert recipe",
]

# --------------- CLIENT ---------------

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

published_after = (
    datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
).isoformat()  # RFC3339, timezone-aware


# --------------- HELPERS ---------------

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def search_food_videos(query, max_results=50, region_code="IN"):
    """
    Returns a list of video IDs for food-related short videos.
    Stops gracefully if quota is exceeded.
    """
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        try:
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                order="viewCount",
                publishedAfter=published_after,
                videoDuration="short",      # <4 min (we'll filter <60s later)
                maxResults=min(50, max_results - len(video_ids)),
                regionCode=region_code,
            )
            response = request.execute()
        except HttpError as e:
            print(f"[ERROR] search() for query '{query}' failed: {e}")
            # If quota exceeded, return what we have so far instead of crashing
            break

        for item in response.get("items", []):
            vid = item["id"]["videoId"]
            video_ids.append(vid)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def get_video_details(video_ids):
    """
    Returns a DataFrame with video metadata and stats for given IDs.
    Handles quota errors gracefully.
    """
    from isodate import parse_duration

    rows = []

    for batch in chunks(video_ids, 50):  # API limit
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch),
            )
            response = request.execute()
        except HttpError as e:
            print(f"[ERROR] videos().list failed: {e}")
            break

        for item in response.get("items", []):
            vid = item["id"]
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            duration_iso = content.get("duration", "PT0S")
            duration_seconds = parse_duration(duration_iso).total_seconds()

            row = {
                "video_id": vid,
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                # tags is a list → json-encode so we can store in CSV cleanly
                "tags": json.dumps(snippet.get("tags", [])),
                "thumbnail_url": snippet.get("thumbnails", {})
                                     .get("high", {})
                                     .get("url"),
                "duration_seconds": duration_seconds,
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "favorite_count": int(stats.get("favoriteCount", 0))
                if "favoriteCount" in stats else 0,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def fetch_transcripts(video_ids):
    """
    Fetch YouTube transcripts where available.
    This does NOT use YouTube Data API quota.
    """
    transcripts = {}
    for vid in tqdm(video_ids, desc="Fetching transcripts"):
        try:
            t = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
            text = " ".join(seg["text"] for seg in t)
            transcripts[vid] = text
        except Exception:
            transcripts[vid] = None
    return transcripts


# --------------- MAIN ---------------

def main():
    all_video_ids = set()

    for q in FOOD_QUERIES:
        print(f"\n=== Query: {q!r} ===")
        vids = search_food_videos(q, max_results=500, region_code="IN")
        print(f"Found {len(vids)} video IDs for query '{q}'")
        all_video_ids.update(vids)

        # If we get 0 for a query, it's very likely quota is gone
        if len(vids) == 0:
            print("Got 0 results; likely quota exceeded. Stopping further queries.")
            break

    all_video_ids = list(all_video_ids)
    print(f"\nTotal unique video IDs collected this run: {len(all_video_ids)}")

    if not all_video_ids:
        print("No videos collected (quota may be fully exhausted). Exiting.")
        return

    # Video details
    videos_df = get_video_details(all_video_ids)
    if videos_df.empty:
        print("No video details could be fetched. Exiting.")
        return

    # Keep only shorts (duration < 60s)
    videos_df = videos_df[videos_df["duration_seconds"] < 60].reset_index(drop=True)
    print(f"Short videos (<60s) after filtering: {len(videos_df)}")

    # Transcripts (optional but useful for NLP)
    transcripts = fetch_transcripts(videos_df["video_id"].tolist())
    videos_df["transcript"] = videos_df["video_id"].map(transcripts)

    # Save / accumulate into CSV
    if os.path.exists(OUTPUT_CSV):
        old_df = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([old_df, videos_df], ignore_index=True)
        combined.drop_duplicates(subset=["video_id"], inplace=True)
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"\nAppended to existing {OUTPUT_CSV}. Total rows now: {len(combined)}")
    else:
        videos_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(videos_df)} rows to new {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

