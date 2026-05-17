# CafeCompass Vancouver: Metadata-Aware Café and Food Spot Recommender

## Project overview

I built CafeCompass Vancouver because I wanted a more personal way to explore cafés, restaurants, and food spots around Vancouver. As an international student at UBC, I spent most of my first year studying in my dorm and did not explore the city as much as I wanted to. This project began as a way to fix that problem with data: instead of only asking what is highly rated nearby, I wanted to ask what fits the experience I am looking for.

The current app is a metadata-aware recommender. It uses place names, cuisine tags, categories, coordinates, and ratings when available to recommend Vancouver food spots based on a selected starting location, desired experience, cuisine filter, distance, and simple ranking preferences.

## Why I built this

I wanted this project to show a different side of data science from forecasting or regression projects. CafeCompass is about ranking and recommendation: taking imperfect local data, turning it into useful features, and building an interface that helps someone make a decision.

The broader idea is still review-aware recommendation. In the future, I want to incorporate permitted review or community text so the model can reason about behavioural and emotional signals such as whether a café feels study-friendly, whether people mention outlets or Wi-Fi, whether a restaurant feels better for date night, or whether a spot is seen as good value.

## What makes it different from Google Maps

Google Maps is very good at answering "What is highly rated near me?" I wanted CafeCompass to move toward a different question: "Which place fits what I want to do?"

In the current version, that means matching a user's selected experience and cuisine preference against place metadata and distance. It is not yet a full behavioural review recommender. I am being intentional about that distinction because a portfolio project should be technically ambitious without pretending that unavailable data exists.

## Current app direction

The Streamlit app is currently focused on a coherent, working recommendation experience:

- Explore Vancouver food places on a map.
- Filter by cuisine, category, rating when available, and location.
- Get recommendations from a selected starting point such as UBC, Downtown, Kitsilano, Mount Pleasant, or Richmond.
- Choose an experience such as quiet study café, cheap eats, date-night restaurant, quick lunch, dessert/matcha, or late-night food.
- Adjust practical ranking tradeoffs such as distance and ratings when ratings exist.
- Find similar places using cuisine, place type, name/category terms, and distance.
- Review data coverage and understand how the ranking works.

I kept the app narrower because that makes it more honest and usable. The current recommender is not pretending to read reviews that are not linked to the Vancouver place table.

## Why the app is not fully review-aware yet

I originally wanted CafeCompass to be review-aware from the start, but the data constraints were more important than forcing the feature into the app.

The biggest issue was Vancouver review coverage. I set up the Yelp Open Dataset path and downloaded the dataset through Kaggle, but the release I tested did not contain Vancouver businesses. That meant it could still be useful as a prototype review-text dataset, but it could not honestly power Vancouver-specific recommendations.

I also looked at other ways to add review or community text, but each option had tradeoffs:

- I did not scrape Google Maps, Yelp pages, TripAdvisor, or similar sites because that would create terms-of-service and reproducibility issues.
- Yelp Fusion API is useful for some live business metadata, but it only provides limited review excerpts, so it is not enough for full NLP-based behavioural recommendation.
- Reddit API text could be useful for local community opinions, but API setup and access were unreliable during development, and Reddit text would still need careful filtering, privacy handling, and place matching.
- Manual community text is possible, but it needs to be sourced carefully, linked to places, and documented before it should affect rankings.

The technical issue is not just collecting text. The hard part is linking noisy text to the correct place, extracting reliable behavioural signals, measuring sentiment around those signals, and showing evidence without fabricating anything. Until that pipeline is populated with trustworthy Vancouver data, I decided the app should remain metadata-aware and clearly say what it can and cannot do.

## Data source and scraping decision

I intentionally chose not to scrape Google Maps, Yelp web pages, TripAdvisor, or similar review platforms directly. Google Maps Platform terms restrict scraping/exporting Maps content, including business names, addresses, and user reviews. Yelp Fusion API is also not suitable as the main review-text source because its official Reviews endpoint returns only a small number of excerpts per business.

Because this is a portfolio project, I wanted the data collection process to be reproducible and ethically safer. The current app therefore relies on permitted/public metadata sources, especially OpenStreetMap and City of Vancouver Open Data. The repository also includes pipeline modules for future review/community-text enrichment through permitted sources.

## Data sources used

- OpenStreetMap / Overpass API: current local place metadata for cafés, restaurants, fast food, food courts, cuisine tags, coordinates, and opening hours.
- City of Vancouver Open Data: official local business and food vendor metadata.
- Yelp Open Dataset: available in the pipeline for review-text experimentation, but the downloaded release did not contain Vancouver businesses.
- Reddit API through PRAW: optional future source for Vancouver-specific community discussion text if API access is available and privacy rules are respected.
- Yelp Fusion API: optional live metadata enrichment only, not a full review-text source.
- Manual community text CSV: optional future fallback for carefully sourced, place-linked notes without usernames.

## Technical stack

- Python for the data pipeline and recommendation logic
- pandas and NumPy for cleaning, joins, feature tables, and numeric scoring
- scikit-learn concepts for similarity scoring, baselines, and evaluation framing
- rapidfuzz for fuzzy place-name matching and entity resolution
- Haversine distance for geospatial filtering and distance-based ranking
- requests for API collection
- Plotly for interactive map visualizations
- Matplotlib for supporting visualizations
- Streamlit for the interactive dashboard
- joblib and NumPy model artifacts for future embedding workflows
- NLTK, spaCy, VADER-style sentiment scoring, and sentence-transformers are included for the planned review-aware NLP extension

## Skills demonstrated

This project demonstrates data cleaning, geospatial filtering, fuzzy matching, entity-resolution design, feature engineering, content-based recommendation, hybrid ranking, baseline comparison, Streamlit dashboarding, and ethical data-source planning.

It also includes scaffolding for NLP work: text preprocessing, aspect extraction, sentiment scoring, sentence embeddings, evidence snippets, and behavioural review signals. Those modules are part of the longer-term direction, but the current app only exposes signals that are actually populated for Vancouver.

## Methodology

The current pipeline starts by collecting Vancouver place metadata from OpenStreetMap and City of Vancouver Open Data. The data is cleaned into a place-level feature table with names, categories, cuisines, coordinates, optional ratings, and source metadata.

The recommender then builds scores from:

- Experience fit: whether the selected experience and cuisine terms match the place name, cuisine, or category metadata.
- Distance fit: how close the place is to the selected starting point using Haversine distance.
- Rating fit: a 1-to-5 star signal when ratings are available.

The final ranking is a weighted combination of those available signals. The selected experience is always treated as the main intent; the user-facing controls only adjust practical tradeoffs such as distance and ratings.

## Recommender architecture

The app currently uses a streamlined hybrid recommender:

1. Filter places by maximum distance and optional cuisine text.
2. Score metadata match against the selected experience.
3. Score distance from the selected starting location.
4. Optionally include ratings when they exist in the data.
5. Rank places by the weighted score and show the top recommendations.

The repository also contains earlier-stage components for distance baselines, rating/popularity baselines, aspect-based recommendation, embeddings, and review-aware scoring. I kept those pieces because they document the intended direction, but I do not expose inactive review signals in the final app.

## Future review-aware extension

The original vision for CafeCompass is still to recommend places based on actual experience described in review or community text. To do that properly, I need a permitted and sufficiently rich Vancouver text source.

Later, I want to add review/community-text signals that capture behavioural and emotional dimensions such as:

- quiet study atmosphere
- laptop friendliness
- outlets and Wi-Fi mentions
- not feeling rushed
- cheap value
- date-night atmosphere
- hidden-gem language
- authentic food
- group friendliness
- service speed
- dessert/matcha/café-hopping intent
- late-night usefulness

Technically, this would use text preprocessing, aspect extraction, sentiment scoring, sentence embeddings, place-level profile aggregation, and explainable evidence snippets. I would only add these signals to the app once the text is legally sourced, linked to places, and reliable enough to improve the recommendations.

## Evaluation approach

Because this project does not yet have real user feedback, I use practical sanity checks rather than pretending there is a perfect ground truth:

- Compare the current hybrid ranking against distance-only and rating-oriented baselines.
- Run synthetic preference queries such as quiet study café near UBC, cheap eats near Downtown, and dessert spot Vancouver.
- Check average distance in top results.
- Check cuisine and category diversity in recommendations.
- Inspect whether changing the desired experience changes the ranked places.

These checks do not prove that the recommender is objectively correct, but they help verify that the ranking logic responds to user intent and does not collapse into only distance.

## Streamlit demo

The Streamlit app includes:

- Project Overview
- Explore Vancouver Food Map
- Café/Food Spot Recommender
- Similar Places
- Data Coverage
- How Ranking Works

I kept the Ranking/How-it-works page because it makes the model transparent. It is useful for a portfolio project: a viewer can see that the recommendations are based on metadata fit, distance, and optional ratings rather than a black-box claim.

## Key insights

The main insight from this stage is that even a metadata-only recommender needs careful framing. It is easy to overstate what a system can infer. If the app only has names, categories, cuisine tags, coordinates, and partial ratings, then the recommendations should be presented as metadata-aware, not review-aware.

That said, the project still demonstrates the shape of a real recommendation system: feature generation, ranking logic, user controls, baselines, geospatial reasoning, and explainability. The next meaningful improvement is not adding more UI polish; it is adding reliable behavioural text data.

## Limitations

- The current app is metadata-aware, not fully review-aware.
- Many OpenStreetMap records do not include ratings or review counts.
- Cuisine and category tags can be incomplete, inconsistent, or too broad.
- Recommendations may not reflect current opening hours, closures, prices, menus, or ownership changes.
- Distance is estimated geographically and does not account for transit time, walking routes, or wait times.
- The Yelp Open Dataset release I downloaded did not include Vancouver businesses, so it cannot currently serve as the main Vancouver review source.
- Reddit/community text would need careful filtering, privacy handling, and place linking before it should influence rankings.
- The app is a portfolio recommender prototype, not a production replacement for Google Maps, Yelp, or local search.

## Future improvements

- Find or build a permitted Vancouver review/community-text source.
- Link behavioural text to places through fuzzy matching and coordinates.
- Add aspect extraction for study-friendly, date-night, cheap-value, hidden-gem, authentic, group-friendly, and late-night signals.
- Add sentiment scoring around each aspect instead of using only overall sentiment.
- Create place embeddings from review/community snippets and compare them with user preference text.
- Add confidence scores so places with little evidence are not over-ranked.
- Improve neighborhood inference from coordinates.
- Add opening-hours freshness, closure checks, and price/menu metadata if reliable sources are available.
- Collect lightweight user feedback to evaluate recommendation relevance.

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the demo dataset:

```bash
python src/run_pipeline.py --demo
```

Build the Vancouver metadata dataset from OpenStreetMap and City of Vancouver Open Data:

```bash
python src/run_pipeline.py --local
```

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Optional review/text experiments are still available in the repository, but they are not required for the current app:

```bash
python src/run_pipeline.py --setup-yelp
python src/run_pipeline.py --yelp
python src/run_pipeline.py --reddit
python src/run_pipeline.py --community-text
python src/run_pipeline.py --yelp-fusion
```

Do not commit raw datasets, processed datasets, model artifacts, API credentials, or `.env` files.
