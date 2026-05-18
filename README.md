# CafeCompass Vancouver: Metadata-Aware Café and Food Spot Recommender

CafeCompass Vancouver is a Streamlit recommender app for exploring cafés, restaurants, and food spots around Vancouver. I built it because, as an international student at UBC, I spent much of my first year studying in my dorm instead of exploring the city. I wanted a project that could help answer a more personal question than "what is highly rated near me?": **what place fits the kind of experience I want right now?**

The current version is intentionally metadata-aware rather than fully review-aware. It recommends places using available Vancouver place metadata: names, cuisine tags, categories, coordinates, and ratings when they exist.

## Current App

The app lets users:

- Explore Vancouver food spots on an interactive map.
- Filter by cuisine, category, location, distance, and rating when available.
- Get recommendations from starting points such as UBC, Downtown Vancouver, Kitsilano, Mount Pleasant, or Richmond.
- Choose experiences like quiet study café, cheap eats, date-night restaurant, quick lunch, dessert/matcha, and late-night food.
- Find similar places based on cuisine, category terms, name similarity, and distance.
- Review data coverage and understand how the ranking works.

The final app is deliberately honest about its current data. It does not claim to understand review behaviour until reliable Vancouver review/community text is legally sourced, linked to places, and validated.

## How The Recommender Works

The recommender uses a streamlined hybrid ranking system:

1. Filter places by maximum distance and optional cuisine text.
2. Score how well each place matches the selected experience using place names, cuisine tags, and categories.
3. Score distance from the selected starting location using Haversine distance.
4. Use ratings only when they are available in the data.
5. Combine the available signals into a final ranked list.

The selected experience is always treated as the main user intent. The app only exposes practical controls a normal user would care about, such as distance preference and rating preference when ratings exist.

## Data Sources

- **OpenStreetMap / Overpass API**: current local place metadata, including cafés, restaurants, fast food, food courts, cuisine tags, coordinates, and opening hours.
- **City of Vancouver Open Data**: official local business and food vendor metadata.
- **Yelp Open Dataset**: included in the pipeline for review-text experimentation, but the downloaded release did not contain Vancouver businesses.
- **Reddit API / manual community text**: planned future sources for community experience signals if access, privacy, and place-linking quality are handled properly.
- **Yelp Fusion API**: optional metadata enrichment only; it is not suitable as the main review-text source because it provides limited review excerpts.

I intentionally avoided scraping Google Maps, Yelp pages, TripAdvisor, or similar platforms. This project is designed to use permitted, reproducible data sources.

## Why It Is Not Fully Review-Aware Yet

The original goal was to build a review-aware recommender that could understand behavioural signals such as "quiet enough to study," "has outlets," "doesn't feel rushed," "good for date night," or "worth the price." In practice, the data constraints were too important to ignore.

The main issue was Vancouver review coverage. I set up the Yelp Open Dataset path and downloaded the dataset through Kaggle, but the release I tested did not include Vancouver businesses. Yelp Fusion was also not enough because it only returns limited review excerpts. Reddit/community text could help later, but it needs careful privacy handling, filtering, and entity resolution before it should influence rankings.

The challenge is not only collecting text. The harder technical problem is linking noisy text to the right place, extracting reliable aspect signals, scoring sentiment around those aspects, and showing evidence without fabricating anything. Until that pipeline is trustworthy, the app remains metadata-aware.

## Technical Skills Demonstrated

This project demonstrates:

- Python data pipeline design
- pandas and NumPy data cleaning
- geospatial filtering with Haversine distance
- fuzzy matching and entity-resolution design
- feature engineering for recommendation systems
- content-based and hybrid ranking logic
- baseline comparison and recommender sanity checks
- Plotly map visualization
- Streamlit dashboard development
- ethical data-source planning

The repository also includes scaffolding for the future NLP direction: text preprocessing, aspect extraction, sentiment scoring, sentence embeddings, evidence snippets, and behavioural review signals.

## Evaluation

Because the app does not yet have real user feedback, I use practical sanity checks:

- Compare hybrid recommendations against distance-only and rating-oriented baselines.
- Run synthetic preference queries such as quiet study café near UBC or cheap eats near Downtown.
- Check average distance in top recommendations.
- Check cuisine and category diversity.
- Confirm that changing the desired experience changes the ranked places.

These checks do not prove that every recommendation is objectively correct, but they verify that the model responds to user intent and does not collapse into only distance.

## Limitations

- The current app is metadata-aware, not fully review-aware.
- Many OpenStreetMap records do not include ratings or review counts.
- Cuisine and category tags can be incomplete or inconsistent.
- Recommendations may not reflect current closures, menus, prices, ownership changes, or wait times.
- Distance is geographic and does not account for transit or walking routes.
- Yelp Open Dataset did not provide Vancouver review coverage in the release I tested.
- Reddit/community text needs careful sourcing and place-linking before it should affect rankings.
- This is a portfolio recommender prototype, not a production replacement for Google Maps, Yelp, or local search.

## Future Improvements

Next, I want to find a reliable way to add permitted review or community text so the recommender can analyse behavioural and emotional signals. That would allow the system to reason about quiet study cafés, laptop friendliness, outlets, Wi-Fi, value, date-night atmosphere, hidden-gem language, authentic food, service speed, and group friendliness.

Technically, the next phase would involve stronger entity resolution, aspect extraction, aspect-level sentiment scoring, sentence embeddings, confidence scores, and evidence snippets. I would only expose those signals in the app once the underlying data is trustworthy enough to improve recommendations rather than just make the project sound more advanced.
