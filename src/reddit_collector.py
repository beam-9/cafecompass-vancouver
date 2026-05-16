from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from config import PROCESSED_DIR, ensure_dirs

SUBREDDITS = ["vancouver", "UBC", "NiceVancouver", "richmondbc"]
QUERIES = [
    "quiet cafe Vancouver",
    "best cafe UBC study",
    "laptop friendly cafe Vancouver",
    "cheap eats Vancouver",
    "hidden gem restaurant Vancouver",
    "date night restaurant Vancouver",
    "best ramen Vancouver",
    "best sushi Vancouver",
    "best matcha Vancouver",
    "late night food Vancouver",
    "group dinner Vancouver",
    "dessert Vancouver",
]


def _credentials_available() -> bool:
    return all(os.getenv(key) for key in ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"])


def collect_reddit_discussions(limit_per_query: int = 25, include_comments: bool = True) -> pd.DataFrame:
    ensure_dirs()
    if not _credentials_available():
        print(
            "Reddit credentials are missing. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, "
            "and REDDIT_USER_AGENT in .env or your shell to enable optional Reddit collection."
        )
        out = pd.DataFrame(
            columns=["post_id", "comment_id", "subreddit", "title", "body", "created_utc", "score", "permalink", "query"]
        )
        out.to_csv(PROCESSED_DIR / "reddit_vancouver_food_discussions.csv", index=False)
        return out

    try:
        import praw
        from dotenv import load_dotenv
    except ImportError:
        print("Install praw and python-dotenv to use the optional Reddit collector.")
        return pd.DataFrame()

    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )

    rows = []
    for subreddit_name in SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)
        for query in QUERIES:
            for post in subreddit.search(query, sort="relevance", time_filter="all", limit=limit_per_query):
                rows.append(
                    {
                        "post_id": post.id,
                        "comment_id": None,
                        "subreddit": subreddit_name,
                        "title": post.title,
                        "body": post.selftext,
                        "created_utc": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
                        "score": post.score,
                        "permalink": f"https://www.reddit.com{post.permalink}",
                        "query": query,
                    }
                )
                if include_comments:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:10]:
                        rows.append(
                            {
                                "post_id": post.id,
                                "comment_id": comment.id,
                                "subreddit": subreddit_name,
                                "title": post.title,
                                "body": comment.body,
                                "created_utc": datetime.fromtimestamp(comment.created_utc, timezone.utc).isoformat(),
                                "score": comment.score,
                                "permalink": f"https://www.reddit.com{comment.permalink}",
                                "query": query,
                            }
                        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["post_id", "comment_id", "body"])
    out.to_csv(PROCESSED_DIR / "reddit_vancouver_food_discussions.csv", index=False)
    print(f"Saved {len(out):,} Reddit discussion rows without usernames.")
    return out


if __name__ == "__main__":
    collect_reddit_discussions()

