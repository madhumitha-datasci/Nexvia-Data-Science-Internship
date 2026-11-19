
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import praw
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    YOUTUBE_API_KEY
)

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

SUBREDDITS = ["technology", "science", "worldnews"]

SEARCH_QUERIES = [
    "artificial intelligence",
    "climate change",
    "technology"
]

REDDIT_POST_LIMIT = 50
REDDIT_COMMENT_LIMIT = 10

YOUTUBE_RESULTS_PER_PAGE = 50
YOUTUBE_MAX_PAGES = 2
YOUTUBE_COMMENT_LIMIT = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# JSON STORAGE
# --------------------------------------------------

def save_json(data, filename):
    """Save collected data as raw JSON."""

    file_path = RAW_DATA_DIR / filename

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    logger.info("Saved data: %s", file_path)

    return file_path


def get_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------
# REDDIT API CONNECTION
# --------------------------------------------------

def connect_reddit():
    """Connect to Reddit using PRAW."""

    if not all([
        REDDIT_CLIENT_ID,
        REDDIT_CLIENT_SECRET,
        REDDIT_USER_AGENT
    ]):
        logger.warning("Reddit credentials are missing.")
        return None

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        logger.info("Reddit API connection initialized.")
        return reddit

    except Exception as error:
        logger.error("Reddit connection failed: %s", error)
        return None


# --------------------------------------------------
# REDDIT POST COLLECTION
# --------------------------------------------------

def collect_reddit_posts(reddit):
    """Collect Reddit posts and their top-level comments."""

    posts = []
    comments = []

    for subreddit_name in SUBREDDITS:

        logger.info("Collecting from r/%s", subreddit_name)

        try:
            subreddit = reddit.subreddit(subreddit_name)

            for submission in subreddit.search(
                " OR ".join(SEARCH_QUERIES),
                sort="new",
                time_filter="month",
                limit=REDDIT_POST_LIMIT
            ):

                author = (
                    str(submission.author)
                    if submission.author else None
                )

                post = {
                    "platform": "reddit",
                    "post_id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "author": author,
                    "subreddit": subreddit_name,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "url": submission.url,
                    "permalink": submission.permalink
                }

                posts.append(post)

                # Collect a limited number of top-level comments
                try:
                    for comment in submission.comments[
                        :REDDIT_COMMENT_LIMIT
                    ]:

                        comment_author = (
                            str(comment.author)
                            if comment.author else None
                        )

                        comments.append({
                            "platform": "reddit",
                            "comment_id": comment.id,
                            "post_id": submission.id,
                            "author": comment_author,
                            "text": comment.body,
                            "score": comment.score,
                            "parent_id": comment.parent_id,
                            "created_utc": comment.created_utc,
                            "subreddit": subreddit_name
                        })

                except Exception as comment_error:
                    logger.warning(
                        "Comment collection failed: %s",
                        comment_error
                    )

            # Avoid unnecessary rapid requests
            time.sleep(2)

        except Exception as error:
            logger.error(
                "Error collecting r/%s: %s",
                subreddit_name,
                error
            )
            continue

    return posts, comments


# --------------------------------------------------
# YOUTUBE API CONNECTION
# --------------------------------------------------

def connect_youtube():
    """Initialize YouTube Data API v3."""

    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key is missing.")
        return None

    try:
        youtube = build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY,
            cache_discovery=False
        )

        logger.info("YouTube API initialized.")
        return youtube

    except Exception as error:
        logger.error("YouTube connection failed: %s", error)
        return None


# --------------------------------------------------
# YOUTUBE COMMENT COLLECTION
# --------------------------------------------------

def collect_youtube_comments(youtube, video_id):

    comments = []

    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=YOUTUBE_COMMENT_LIMIT,
            textFormat="plainText",
            order="relevance"
        ).execute()

        for item in response.get("items", []):

            comment = item["snippet"]["topLevelComment"]
            details = comment["snippet"]

            comments.append({
                "platform": "youtube",
                "comment_id": comment["id"],
                "video_id": video_id,
                "author": details.get("authorDisplayName"),
                "author_channel_id": (
                    details.get("authorChannelId", {})
                    .get("value")
                ),
                "text": details.get("textDisplay"),
                "like_count": details.get("likeCount", 0),
                "published_at": details.get("publishedAt")
            })

    except HttpError as error:
        logger.warning(
            "Comments unavailable for video %s: %s",
            video_id,
            error
        )

    except Exception as error:
        logger.warning(
            "Comment collection failed for %s: %s",
            video_id,
            error
        )

    return comments


# --------------------------------------------------
# YOUTUBE VIDEO COLLECTION
# --------------------------------------------------

def collect_youtube_videos(youtube):

    videos = []
    comments = []

    for query in SEARCH_QUERIES:

        logger.info("Searching YouTube for: %s", query)

        next_page_token = None

        for page in range(YOUTUBE_MAX_PAGES):

            try:
                search_response = youtube.search().list(
                    part="snippet",
                    q=query,
                    type="video",
                    order="date",
                    maxResults=YOUTUBE_RESULTS_PER_PAGE,
                    pageToken=next_page_token
                ).execute()

                video_ids = []

                for item in search_response.get("items", []):

                    video_id = item["id"]["videoId"]
                    video_ids.append(video_id)

                if not video_ids:
                    break

                # Fetch video details and statistics
                details_response = youtube.videos().list(
                    part="snippet,statistics",
                    id=",".join(video_ids)
                ).execute()

                for item in details_response.get("items", []):

                    snippet = item.get("snippet", {})
                    statistics = item.get("statistics", {})

                    video = {
                        "platform": "youtube",
                        "video_id": item["id"],
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "channel_id": snippet.get("channelId"),
                        "channel_title": snippet.get("channelTitle"),
                        "published_at": snippet.get("publishedAt"),
                        "view_count": statistics.get("viewCount"),
                        "like_count": statistics.get("likeCount"),
                        "comment_count": statistics.get("commentCount"),
                        "search_query": query
                    }

                    videos.append(video)

                    # Collect comments for interaction analysis
                    video_comments = collect_youtube_comments(
                        youtube,
                        item["id"]
                    )

                    comments.extend(video_comments)

                next_page_token = search_response.get(
                    "nextPageToken"
                )

                if not next_page_token:
                    break

                time.sleep(1)

            except HttpError as error:
                logger.error(
                    "YouTube API error for '%s': %s",
                    query,
                    error
                )
                break

            except Exception as error:
                logger.error(
                    "YouTube collection failed: %s",
                    error
                )
                break

    return videos, comments


# --------------------------------------------------
# MAIN DATA COLLECTION PIPELINE
# --------------------------------------------------

def main():

    timestamp = get_timestamp()

    reddit_posts = []
    reddit_comments = []

    youtube_videos = []
    youtube_comments = []

    logger.info("Starting DS-03 data collection.")

    # Reddit
    reddit = connect_reddit()

    if reddit:
        reddit_posts, reddit_comments = (
            collect_reddit_posts(reddit)
        )

    # YouTube
    youtube = connect_youtube()

    if youtube:
        youtube_videos, youtube_comments = (
            collect_youtube_videos(youtube)
        )

    # Save original collected data
    save_json(
        reddit_posts,
        f"reddit_posts_{timestamp}.json"
    )

    save_json(
        reddit_comments,
        f"reddit_comments_{timestamp}.json"
    )

    save_json(
        youtube_videos,
        f"youtube_videos_{timestamp}.json"
    )

    save_json(
        youtube_comments,
        f"youtube_comments_{timestamp}.json"
    )

    # Collection summary
    summary = {
        "collection_time": timestamp,
        "reddit_posts": len(reddit_posts),
        "reddit_comments": len(reddit_comments),
        "youtube_videos": len(youtube_videos),
        "youtube_comments": len(youtube_comments),
        "total_records": (
            len(reddit_posts)
            + len(reddit_comments)
            + len(youtube_videos)
            + len(youtube_comments)
        )
    }

    save_json(
        summary,
        f"collection_summary_{timestamp}.json"
    )

    logger.info("Collection completed.")
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
