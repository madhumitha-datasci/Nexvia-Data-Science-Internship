
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# YouTube API credentials
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def check_api_credentials():
    """Check whether API credentials are configured."""

    reddit_ready = bool(
        REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
    )

    youtube_ready = bool(YOUTUBE_API_KEY)

    print("Reddit API:", "Configured" if reddit_ready else "Not configured")
    print("YouTube API:", "Configured" if youtube_ready else "Not configured")


if __name__ == "__main__":
    check_api_credentials()
