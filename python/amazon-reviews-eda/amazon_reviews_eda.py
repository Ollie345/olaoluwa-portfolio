# =============================================================================
# Amazon Reviews Exploratory Data Analysis (EDA)
# =============================================================================

# --- Imports ---
import json
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download required NLTK language data
# These are needed for stop word removal and lemmatisation
nltk.download('stopwords', force=True)
nltk.download('wordnet')
nltk.download('punkt_tab')


# =============================================================================
# STEP 1 - EXTRACT: Load the dataset
# =============================================================================

def load_dataset(filepath, max_rows=50000):
    """
    Loads the Amazon reviews dataset from a JSON file.
    Each line in the file is a separate JSON object (one review per line).
    max_rows limits how many lines are read to avoid memory issues.
    """
    records = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Stop reading after max_rows to keep memory usage low
            if i >= max_rows:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip any malformed lines
                    pass

    df = pd.DataFrame(records)
    print(f"Dataset loaded: {len(df)} total reviews")
    print(f"Columns: {list(df.columns)}")
    return df


# Load the dataset
df_raw = load_dataset('Industrial_and_Scientific_170.json')  # review data, Industrial and Scientific category (not included in this repo)


# =============================================================================
# STEP 2 - CLEAN: Handle missing data and prepare the dataframe
# =============================================================================

def clean_dataset(df):
    """
    Cleans the raw reviews dataframe by:
    - Dropping rows with missing essential fields
    - Converting data types
    - Parsing dates
    - Removing duplicate reviews
    """
    print("\n--- Data Cleaning ---")
    print(f"Rows before cleaning: {len(df)}")

    # Drop rows where review text, rating or product ID is missing
    df = df.dropna(subset=['reviewText', 'overall', 'asin'])

    # Convert rating to numeric in case it was stored as a string
    df['overall'] = pd.to_numeric(df['overall'], errors='coerce')

    # Convert unix timestamp to a readable datetime column
    df['reviewDate'] = pd.to_datetime(df['unixReviewTime'], unit='s', errors='coerce')

    # Extract year and month separately for time-based analysis
    df['year'] = df['reviewDate'].dt.year
    df['month'] = df['reviewDate'].dt.month

    # Convert helpful votes to numeric - sometimes stored as "1,234" with a comma
    df['vote'] = df['vote'].astype(str).str.replace(',', '', regex=False)
    df['vote'] = pd.to_numeric(df['vote'], errors='coerce').fillna(0).astype(int)

    # Remove duplicate reviews - same reviewer reviewing the same product twice
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['reviewerID', 'asin'])
    print(f"Duplicates removed: {before_dedup - len(df)}")

    print(f"Rows after cleaning: {len(df)}")
    return df.reset_index(drop=True)


# Clean the dataset
df_clean = clean_dataset(df_raw)

# Show how many unique products are in the dataset
print(f"Total unique products: {df_clean['asin'].nunique()}")
print("\nAll products and their review counts:")
print(df_clean['asin'].value_counts())


# =============================================================================
# STEP 3 - SELECT: Choose 3 products to analyse
# =============================================================================

# Manually selected 3 products with a good number of reviews
# These were chosen from the top products by review count
# B00020BQGK = Vacuum cleaner
# B000087KUA = Scotch tape
# B0001GAYRC = Gorilla Glue
selected_asins = ['B00020BQGK', 'B000087KUA', 'B0001GAYRC']

print("\n--- Selected Products ---")
for asin in selected_asins:
    count = df_clean[df_clean['asin'] == asin].shape[0]
    print(f"  ASIN: {asin} | Reviews: {count}")


# =============================================================================
# STEP 4 - NLP: Text preprocessing and top word extraction
# =============================================================================

def preprocess_text(text):
    """
    Cleans and normalises a single review text string.
    Steps:
    1. Lowercase the text
    2. Remove punctuation and numbers
    3. Split into individual words (tokens)
    4. Remove stop words (common words like 'the', 'and', 'is')
    5. Lemmatise - reduce words to base form e.g. 'running' -> 'run'
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # Split into words
    tokens = text.split()

    # Remove stop words and very short words (less than 3 characters)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Lemmatise each word to its base form
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def get_top_words(df_product, n=20):
    """
    Returns the most frequent words across all reviews for a product.
    Uses preprocess_text to clean each review before counting.
    """
    all_tokens = []
    for text in df_product['reviewText'].dropna():
        all_tokens.extend(preprocess_text(str(text)))
    return Counter(all_tokens).most_common(n)


# Print top 20 words for each selected product
print("\n--- Top Words for each product ---")
for asin in selected_asins:
    df_product = df_clean[df_clean['asin'] == asin]
    print(f"\nASIN: {asin}")
    for word, count in get_top_words(df_product):
        print(f"  {word}: {count}")


# =============================================================================
# STEP 5 - EDA: Summary statistics per product
# =============================================================================

def print_summary_stats(df_product, asin):
    """
    Prints key summary statistics for a single product's reviews including:
    - Total reviews, average rating, standard deviation
    - Percentage of 5-star and 1-star reviews
    - Average helpful votes
    - Date range of reviews
    """
    print(f"\n{'='*60}")
    print(f"Product ASIN: {asin}")
    print(f"{'='*60}")
    print(f"  Total reviews     : {len(df_product)}")
    print(f"  Average rating    : {df_product['overall'].mean():.2f} / 5.0")
    print(f"  Rating std dev    : {df_product['overall'].std():.2f}")
    print(f"  5-star reviews    : {(df_product['overall'] == 5).sum()} "
          f"({(df_product['overall'] == 5).mean()*100:.1f}%)")
    print(f"  1-star reviews    : {(df_product['overall'] == 1).sum()} "
          f"({(df_product['overall'] == 1).mean()*100:.1f}%)")
    print(f"  Avg helpful votes : {df_product['vote'].mean():.2f}")
    print(f"  Date range        : {df_product['reviewDate'].min().date()} "
          f"to {df_product['reviewDate'].max().date()}")


# Print summary statistics for each selected product
for asin in selected_asins:
    df_product = df_clean[df_clean['asin'] == asin]
    print_summary_stats(df_product, asin)


# =============================================================================
# STEP 6 - VISUALISATIONS
# =============================================================================

def plot_rating_distribution(df_product, asin):
    """
    Produces a bar chart showing how ratings are distributed for a product.
    Each bar represents a star rating (1-5) and its height shows the count.
    Saved as a PNG file.
    """
    rating_counts = df_product['overall'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(rating_counts.index.astype(int), rating_counts.values,
                  color='steelblue', edgecolor='white', width=0.6)

    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                str(int(height)), ha='center', va='bottom', fontsize=10)

    ax.set_title(f"Rating Distribution - ASIN: {asin}", fontsize=12)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f"rating_{asin}.png")
    plt.show()


# Plot rating distribution for each product
for asin in selected_asins:
    df_product = df_clean[df_clean['asin'] == asin]
    plot_rating_distribution(df_product, asin)


def plot_reviews_over_time(df_product, asin):
    """
    Produces a line chart showing the number of reviews per month over time.
    This reveals seasonal trends or spikes in customer interest.
    Saved as a PNG file.
    """
    df_product = df_product.copy()

    # Group reviews by year and month
    df_product['yearmonth'] = df_product['reviewDate'].dt.to_period('M').dt.to_timestamp()
    monthly = df_product.groupby('yearmonth').size().reset_index(name='count')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly['yearmonth'], monthly['count'],
            color='steelblue', linewidth=1.8, marker='o', markersize=3)

    # Shaded area under the line for visual clarity
    ax.fill_between(monthly['yearmonth'], monthly['count'], alpha=0.15, color='steelblue')

    ax.set_title(f"Reviews Over Time - ASIN: {asin}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Reviews")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"timeline_{asin}.png")
    plt.show()


# Plot reviews over time for each product
for asin in selected_asins:
    df_product = df_clean[df_clean['asin'] == asin]
    plot_reviews_over_time(df_product, asin)


def plot_wordcloud(df_product, asin):
    """
    Generates a word cloud from the most frequent words in reviews.
    Word size represents how frequently that word appears.
    Uses preprocessed tokens (stop words removed, lemmatised).
    Saved as a PNG file.
    """
    all_tokens = []
    for text in df_product['reviewText'].dropna():
        all_tokens.extend(preprocess_text(str(text)))

    # Join tokens into a single string for the WordCloud library
    text_blob = ' '.join(all_tokens)

    # Generate the word cloud
    wc = WordCloud(width=800, height=400,
                   background_color='white',
                   colormap='Blues',
                   max_words=100).generate(text_blob)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Most Common Words - ASIN: {asin}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"wordcloud_{asin}.png")
    plt.show()


# Plot word cloud for each product
for asin in selected_asins:
    df_product = df_clean[df_clean['asin'] == asin]
    plot_wordcloud(df_product, asin)


def plot_bubble_chart(df_product, asin):
    """
    Produces a bubble chart showing average rating per year.
    Bubble size represents total helpful votes received that year.
    Bubble colour represents number of reviews that year.
    This reveals whether product quality perception changes over time.
    Saved as a PNG file.
    """
    # Aggregate data by year
    yearly = df_product.groupby('year').agg(
        avg_rating=('overall', 'mean'),
        review_count=('overall', 'count'),
        total_votes=('vote', 'sum')
    ).reset_index()

    # Scale bubble size for readability, minimum size of 10
    bubble_size = yearly['total_votes'].apply(lambda x: max(x, 10)) * 3

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(yearly['year'], yearly['avg_rating'],
                         s=bubble_size, alpha=0.6,
                         c=yearly['review_count'], cmap='YlOrRd',
                         edgecolors='gray', linewidth=0.5)

    # Annotate each bubble with the number of reviews that year
    for _, row in yearly.iterrows():
        ax.annotate(f"n={int(row['review_count'])}",
                    (row['year'], row['avg_rating']),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color='gray')

    plt.colorbar(scatter, ax=ax, label='Review count')
    ax.set_title(f"Avg Rating vs Year - ASIN: {asin}", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Rating")
    ax.set_ylim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(f"bubble_{asin}.png")
    plt.show()


# Plot bubble chart for each product
for asin in selected_asins:
    df_product = df_clean[df_clean['asin'] == asin]
    plot_bubble_chart(df_product, asin)

print("\n--- Analysis complete ---")
print("All charts saved as PNG files in the current directory.")
