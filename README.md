# CafeCompass Vancouver: Review-Aware Café and Food Spot Recommender

## Project overview

I built CafeCompass Vancouver because I wanted to create a recommender system that goes beyond ratings and distance. When I search for cafes, I usually care about things that are hidden inside reviews, such as whether a place is quiet enough to study, whether there are outlets, whether people feel rushed, or whether the food is actually worth the price.

This project uses NLP and recommendation logic to turn that kind of unstructured review and community text into a more personalized recommendation experience for Vancouver cafes, restaurants, and food spots.

## Why I built this

I wanted this project to show a different side of data science from my previous forecasting and regression projects. Instead of predicting one numeric outcome, CafeCompass Vancouver focuses on ranking and recommendation. The main challenge was turning messy review text into structured signals that could help answer more personal questions, like whether a cafe is actually good for studying or whether a restaurant feels more like a date-night spot than a quick lunch place.

## What makes it different from Google Maps

Google Maps is very good at helping users find highly rated nearby places. I wanted CafeCompass Vancouver to answer a slightly different question: "Which place fits the experience I want?"

For example, a user might want a quiet cafe near UBC with laptop seating, outlets, good coffee, and reviews that suggest people can stay for a while without feeling rushed. Instead of only using rating and distance, this project extracts experience-based signals from review text and community discussions.

## Data source and scraping decision

I intentionally chose not to scrape Google Maps, Yelp web pages, TripAdvisor, or similar review platforms directly. Google Maps Platform terms restrict scraping/exporting Maps content, including business names, addresses, and user reviews. Yelp Fusion API is also not used as the main review-text source because the official Reviews endpoint only returns up to three review excerpts per business.

Because this is a portfolio project, I wanted the data collection process to be reproducible and ethically safer. I therefore use permitted/public sources such as the Yelp Open Dataset, Reddit API, OpenStreetMap, City of Vancouver Open Data, and optional Foursquare OS Places.

## Data sources used

- Yelp Open Dataset: main review-text source if Vancouver coverage exists.
- OpenStreetMap / Overpass API: current local place metadata for cafes, restaurants, fast food, food courts, cuisine tags, coordinates, and opening hours.
- City of Vancouver Open Data: official business licence and food vendor metadata.
- Reddit API through PRAW: optional Vancouver-specific community discussion text.
- Foursquare OS Places: optional POI/category enrichment if practical.
- Yelp Fusion API: optional live metadata or review excerpt enrichment only, not the main NLP source.

## Technical stack

- Python for the main data pipeline
- pandas and NumPy for data cleaning and transformation
- scikit-learn for similarity scoring, baselines, and evaluation
- sentence-transformers for semantic embeddings
- NLTK/spaCy for review text preprocessing
- VADER or a simple lexicon approach for sentiment scoring
- rapidfuzz for fuzzy business-name matching
- geopy/Haversine distance for geospatial filtering
- Plotly and Matplotlib for visualizations
- Streamlit for the interactive recommender dashboard
- PRAW for optional Reddit API collection
- OpenStreetMap/Overpass API and Vancouver Open Data for local place metadata

## Skills demonstrated

This project demonstrates NLP, text preprocessing, aspect extraction, sentiment scoring, sentence embeddings, geospatial filtering, fuzzy matching, recommendation ranking, baseline evaluation, and Streamlit dashboarding.

## Methodology

The pipeline starts by collecting or loading permitted data sources, then cleaning place metadata and text. Yelp reviews and tips are used when available, Reddit text is optional, and OSM/Vancouver Open Data help improve local coverage. I then extract aspect mentions from text, score sentiment, create place-level profiles, build embeddings, and rank places based on the experience a user asks for.

## Recommender architecture

CafeCompass includes four ranking approaches:

1. A distance baseline that ranks only by distance.
2. A rating/popularity baseline that ranks by stars and review count.
3. An aspect-based recommender that matches user preferences to place aspect scores.
4. A hybrid recommender that combines aspect match, semantic similarity, distance, rating, confidence, context match, and hidden-gem signals.

The hybrid recommender is the main model because it reflects the way I actually make food and cafe decisions: I care about convenience, but I also care about what people say the place feels like.

## NLP aspect extraction

The aspect extraction step looks for experience categories such as quiet study, laptop-friendly, cheap value, hidden gem, authentic food, group-friendly, service speed, food quality, dessert/drinks, and late-night food. For each aspect, the project keeps evidence snippets from the original text so recommendations can explain why a place matched.

## Evaluation approach

Because this project may not have real user feedback, I use several practical evaluation methods:

- Compare the hybrid recommender against distance-only, rating-only, and popularity baselines.
- Use historical text holdout when review dates are available.
- Run synthetic preference tests such as "quiet study cafe near UBC" or "hidden gem ramen Vancouver."
- Measure category and neighborhood diversity in the top results.
- Track explanation coverage, or how often recommendations have direct evidence snippets.

## Streamlit demo

The Streamlit app is designed as an interactive portfolio demo with pages for project overview, map exploration, recommendations, similar places, review intelligence, and model evaluation. A small synthetic demo dataset is included in `data/sample/` so the dashboard can show the intended recommender behavior before I download large external datasets.

## Key insights

The main insight behind this project is that reviews often contain the real decision-making signals that star ratings flatten. Two places can both have strong ratings, but one might be better for studying, another for a date, and another for a quick cheap lunch. By extracting those signals, the recommender can explain matches in a way that feels closer to how people actually choose where to go.

## Limitations

- The Yelp Open Dataset may not include the newest Vancouver restaurants or cafes.
- Vancouver coverage in the Yelp Open Dataset should be checked before relying on it as the main review source.
- Yelp Fusion API is not used as the main review-text source because it only returns limited review excerpts.
- Reddit text can be noisy and may overrepresent opinions from active Reddit users.
- Review text is subjective and can contain bias.
- Some places may have very few reviews, so confidence scores are needed.
- Recommendations may not reflect current opening hours, closures, prices, menus, or ownership changes.
- The project is a portfolio recommender system, not a production replacement for Google Maps or Yelp.

The recommendations depend heavily on the available text data. Places with few reviews may have lower confidence, Reddit comments may be noisy, and the Yelp Open Dataset may not include the newest Vancouver businesses. The app should therefore be treated as a portfolio recommender prototype, not a real-time commercial food discovery platform.

## Future improvements

- Add a lightweight user feedback loop for relevance judgments.
- Improve entity resolution with stronger address parsing.
- Add neighborhood inference from coordinates.
- Use topic modeling to discover new experience themes automatically.
- Add freshness weighting so recent reviews matter more.
- Add better support for temporary closures and current opening hours.

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place Yelp Open Dataset files in:

```text
data/raw/yelp/business.json
data/raw/yelp/review.json
data/raw/yelp/tip.json
data/raw/yelp/checkin.json
```

Run the pipeline modules as needed:

```bash
python src/run_pipeline.py --demo
```

That command builds local ignored files in `data/processed/` from the committed sample dataset. For the Yelp Open Dataset path, configure Kaggle credentials with `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`, then run:

```bash
python src/run_pipeline.py --setup-yelp
python src/run_pipeline.py --yelp
```

Yelp review text is used for behavioural and emotional experience signals. Yelp star ratings are preserved for analysis, but they are not a default ranking signal at this stage.

Individual stages can also be run directly:

```bash
python src/load_yelp_data.py
python src/filter_vancouver.py
python src/clean_reviews.py
python src/osm_collector.py
python src/vancouver_open_data.py
python src/aspect_extraction.py
python src/sentiment_scoring.py
python src/feature_engineering.py
python src/evaluation.py
```

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```
