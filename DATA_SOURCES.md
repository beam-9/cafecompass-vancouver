# Data Sources

I designed this project around permitted/public data sources and official APIs. I do not scrape Google Maps, Yelp web pages, TripAdvisor, or similar review-site pages directly.

| Source | Link | Used for | Credentials | Required | Notes and limitations |
| --- | --- | --- | --- | --- | --- |
| Yelp Open Dataset | https://business.yelp.com/data/resources/open-dataset/ | Main review-text source if Vancouver coverage exists. Uses `business.json`, `review.json`, `tip.json`, and `checkin.json`. | Manual download from Yelp or Kaggle. | Required for full Yelp review NLP, but pipeline can run without it. | Do not commit raw files. Vancouver coverage must be inspected before relying on it. |
| Kaggle Yelp Dataset mirror | https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset | API download path for the Yelp Open Dataset through `python src/run_pipeline.py --setup-yelp`. | Kaggle account/API credentials. | Recommended for setup. | Same usage restrictions as the dataset source. Do not commit raw files. |
| Reddit API | https://redditinc.com/policies/data-api-terms | Optional Vancouver-specific community discussion source from r/vancouver, r/UBC, r/NiceVancouver, and r/richmondbc. | Requires `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT`. | Optional. | Use PRAW. Do not store or display usernames. Reddit text can be noisy and not representative. |
| OpenStreetMap / Overpass API | https://overpass-turbo.eu/ | Current local POI metadata for cafes, restaurants, fast food, food courts, cuisine tags, coordinates, and opening hours. | No credentials. | Recommended. | No review text. Useful for Vancouver-specific coverage and metadata. |
| City of Vancouver Open Data: Business Licences | https://opendata.vancouver.ca/explore/dataset/business-licences/ | Official business licence metadata for local validation and enrichment. | No credentials. | Recommended. | Not a review source. Business category fields may need cleaning. |
| City of Vancouver Open Data: Food Vendors | https://opendata.vancouver.ca/explore/dataset/food-vendors/api/ | Official food vendor metadata. | No credentials. | Recommended. | Not a review source. Coverage differs from restaurants/cafes. |
| Foursquare OS Places | https://opensource.foursquare.com/os-places/ | Optional POI/category enrichment if practical. | Depends on access path. | Optional. | Useful for categories and POI coverage, but not required for the core recommender. |
| Hugging Face Foursquare OS Places | https://huggingface.co/datasets/foursquare/fsq-os-places | Optional dataset access path for Foursquare OS Places. | Hugging Face access may be needed. | Optional. | Do not commit large raw files. |
| Yelp Fusion API | https://docs.developer.yelp.com/docs/fusion-intro | Optional live metadata or review excerpt enrichment. | Yelp API key. | Optional. | Not suitable for full review NLP because the Reviews endpoint returns limited review excerpts. |
