"""
Amazon Reviews EDA - Approach 2 (simple functions + pandas shortcut).

This script does the EXACT same thing as eda_new.py but uses a different
coding style:
  - pd.read_json() loads the compressed file in ONE line (no gzip/json loop).
  - Cleaning uses pandas methods (dropna, between) instead of a Python loop.
  - Each chart has its own small function so the code is easy to reuse.
  - matplotlib draws the charts instead of plotly.

---------------------------------------------------------------------------
HOW TO SET UP A VIRTUAL ENVIRONMENT (venv) AND INSTALL PACKAGES
---------------------------------------------------------------------------
This script needs: pandas, matplotlib, nltk, wordcloud.

1. Open Terminal (macOS/Linux) or Command Prompt / PowerShell (Windows).

2. Go to the folder that contains this file, for example:
       cd path/to/this/folder

3. Create a virtual environment named "venv":
       macOS / Linux:   python3 -m venv venv
       Windows:         py -m venv venv

4. Activate the virtual environment (do this every new terminal session):
       macOS / Linux:   source venv/bin/activate
       Windows (cmd):   venv\\Scripts\\activate.bat
       Windows (PS):    venv\\Scripts\\Activate.ps1

5. Install the packages this script needs:
       pip install pandas matplotlib nltk wordcloud

6. Run the script (make sure the .json.gz file is in the same folder):
       python eda_approach_2.py

7. When you are finished, leave the venv:
       deactivate
---------------------------------------------------------------------------
"""

# ---- IMPORTS ----
# re               : lets us remove punctuation/numbers from text using patterns
# Counter          : counts how many times each word appears (like a tally)
# pandas           : gives us DataFrames and can read JSON files directly
# matplotlib.pyplot: lets us draw charts (bar, line, scatter/bubble)
# nltk             : Natural Language Processing toolkit (stop words + lemmatiser)
# WordCloud        : generates word-cloud images from text
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download the NLTK data files we need (only downloads once, then skips)
# "stopwords" = common words like "the", "is", "and" that add no meaning
# "wordnet"   = dictionary used by the lemmatiser to find word roots
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Create the lemmatiser object once so we can reuse it in any function
lemmatizer = WordNetLemmatizer()

# Load the English stop-words list into a set (sets are faster for lookups)
stop_words = set(stopwords.words("english"))


# ==========================================================================
# FUNCTION: load_reviews
# Opens the compressed .json.gz file and returns a pandas DataFrame.
# Unlike Approach 1, we do NOT need gzip or json because pandas can
# handle both the decompression and the JSON parsing for us in one line.
# ==========================================================================
def load_reviews(filepath):
    """
    Reads a .json.gz file where each line is one JSON review.
    Returns all the reviews as a pandas DataFrame.
    """
    try:
        # pd.read_json does everything in one step:
        #   - compression="gzip" tells pandas the file is compressed
        #   - lines=True tells pandas each line is a separate JSON object
        df = pd.read_json(filepath, lines=True, compression="gzip")
        print(f"Loaded {len(df)} reviews from file.")
        return df

    except FileNotFoundError:
        print("ERROR: File not found. Put the .json.gz file in the same folder.")
        return pd.DataFrame()  # return an empty DataFrame so the script can stop safely

    except Exception as e:
        print(f"ERROR loading file: {e}")
        return pd.DataFrame()


# ==========================================================================
# FUNCTION: clean_reviews
# Takes the raw DataFrame and returns a cleaned version.
# Uses pandas methods instead of a Python for-loop.
# ==========================================================================
def clean_reviews(df):
    """
    Cleans the raw reviews DataFrame:
      1. Drops rows missing essential columns.
      2. Fills in defaults for optional columns.
      3. Cleans the vote column (remove commas, convert to int).
      4. Keeps only ratings between 1 and 5.
      5. Parses dates, extracts year/month, removes duplicates.
    Returns the cleaned DataFrame.
    """

    # ---- 1. Drop rows where essential columns are missing ----
    # subset= lists the columns that MUST have a value
    df = df.dropna(subset=["asin", "reviewText", "overall", "reviewTime"])

    # ---- 2. Fill defaults for optional columns ----
    # .fillna() replaces NaN (missing) values with whatever we give it
    df["reviewerID"] = df["reviewerID"].fillna("Unknown")
    df["reviewerName"] = df["reviewerName"].fillna("Unknown")
    df["summary"] = df["summary"].fillna("")

    # ---- 3. Clean the helpful-votes ("vote") column ----
    # Fill missing votes with "0", remove commas, convert to integer
    df["vote"] = df["vote"].fillna("0")
    df["vote"] = df["vote"].astype(str).str.replace(",", "", regex=False)
    df["vote"] = pd.to_numeric(df["vote"], errors="coerce").fillna(0).astype(int)

    # ---- 4. Keep only ratings between 1 and 5 ----
    # .between(1, 5) returns True/False for each row
    df = df[df["overall"].between(1, 5)]
    # Convert rating to integer (e.g. 5.0 -> 5)
    df["overall"] = df["overall"].astype(int)

    # ---- 5. Parse dates and extract year / month ----
    df["reviewTime"] = pd.to_datetime(df["reviewTime"], errors="coerce")
    df = df.dropna(subset=["reviewTime"])  # drop rows where date parsing failed

    df["year"] = df["reviewTime"].dt.year
    df["month"] = df["reviewTime"].dt.month

    # ---- 6. Remove duplicate reviews (same reviewer + same product) ----
    df = df.drop_duplicates(subset=["reviewerID", "asin"])

    # ---- 7. Rename columns to simpler names ----
    df = df.rename(columns={
        "asin": "asin",
        "reviewerID": "reviewer_id",
        "reviewerName": "reviewer_name",
        "overall": "rating",
        "reviewText": "review_text",
        "summary": "summary",
        "vote": "helpful_votes",
        "reviewTime": "review_date",
    })

    # Keep only the columns we need (drop any extras like "style", "image", etc.)
    keep_cols = [
        "asin", "reviewer_id", "reviewer_name", "rating",
        "review_text", "summary", "helpful_votes", "review_date",
        "year", "month",
    ]
    df = df[keep_cols]

    # Reset the row numbers so they go 0, 1, 2, ... with no gaps
    df = df.reset_index(drop=True)

    print(f"Cleaning complete. {len(df)} reviews kept.")
    return df


# ==========================================================================
# FUNCTION: check_quality
# Runs data quality checks and prints the results.
# Returns True if the data is OK to analyse, False if it is empty.
# ==========================================================================
def check_quality(df):
    """
    Checks the DataFrame for common data issues:
      - Is it empty?
      - Any duplicate rows?
      - Any null values?
      - Any ratings outside 1-5?
    Returns True if data is OK, False if empty.
    """
    print("\n--- RUNNING DATA QUALITY CHECKS ---")

    # Check 1: Is the DataFrame empty?
    if df.empty:
        print("ERROR: DataFrame is empty. Nothing to analyse.")
        return False

    # Check 2: Duplicate reviewer + product?
    has_dupes = df.duplicated(subset=["reviewer_id", "asin"]).any()
    if has_dupes:
        print("WARNING: Duplicate reviewer/product combinations found.")
    else:
        print("PASS: No duplicate entries found.")

    # Check 3: Any null values left?
    if df.isnull().values.any():
        print("WARNING: Null values found.")
        print(df.isnull().sum())
    else:
        print("PASS: No null values found.")

    # Check 4: Ratings all between 1 and 5?
    bad = df[~df["rating"].between(1, 5)]
    if not bad.empty:
        print(f"WARNING: {len(bad)} rows with invalid ratings.")
    else:
        print("PASS: All ratings are valid (1-5).")

    print("Data quality checks complete.\n")
    return True


# ==========================================================================
# FUNCTION: pick_products
# Finds the products with the most reviews so we have enough data to analyse.
# ==========================================================================
def pick_products(df, min_reviews=30, how_many=3):
    """
    Returns a list of product IDs (ASINs) that have at least min_reviews
    reviews.  Picks the top 'how_many' by review count.
    """
    # value_counts() counts reviews per product, sorted highest first
    counts = df["asin"].value_counts()

    # Keep only products with enough reviews
    eligible = counts[counts >= min_reviews]

    # Take the top ones
    selected = eligible.head(how_many).index.tolist()

    print("--- PRODUCTS SELECTED FOR ANALYSIS ---")
    for asin in selected:
        print(f"  ASIN: {asin}  |  Reviews: {counts[asin]}")

    return selected


# ==========================================================================
# FUNCTION: process_review_text
# Applies NLP techniques to the review text for one product:
#   1. Combines all reviews into one string
#   2. Lowercases and removes punctuation/numbers
#   3. Removes stop words (common words like "the", "is", "and")
#   4. Lemmatises each word (converts to base form, e.g. "running" -> "run")
#   5. Counts word frequencies
# Returns the cleaned word list and top 15 (word, count) pairs.
# ==========================================================================
def process_review_text(product_df):
    """
    Takes a product's DataFrame, processes review text with NLP,
    and returns (cleaned_words_list, top_15_pairs).
    """
    # 1. Join every review's text into one big string
    all_text = " ".join(product_df["review_text"].tolist())

    # 2. Convert to lowercase so "Good" and "good" count as the same word
    all_text = all_text.lower()

    # 3. Remove anything that is NOT a letter or space (punctuation, numbers)
    #    re.sub replaces matches with ""; [^a-z\s] means "not a-z and not space"
    all_text = re.sub(r"[^a-z\s]", "", all_text)

    # 4. Split the big string into a list of individual words
    words = all_text.split()

    # 5. Remove stop words using a list comprehension
    words = [w for w in words if w not in stop_words]

    # 6. Lemmatise each word (reduce to base/root form)
    words = [lemmatizer.lemmatize(w) for w in words]

    # 7. Count how often each word appears
    word_counts = Counter(words)

    # 8. Get the 15 most common words
    top_15 = word_counts.most_common(15)

    return words, top_15


# ==========================================================================
# FUNCTION: show_all_charts
# Draws all 6 charts for one product in a single 3x2 window.
# This way you see every chart at once and only need to close one window
# before the next product appears.
# ==========================================================================
def show_all_charts(product_df, asin, words, top_15):
    """
    Creates a 3x2 grid of charts in one window for a single product.
    """

    # fig  = the whole window
    # axes = a 3x2 grid; axes[row][col]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Give the whole window a title with the product ID
    fig.suptitle(f"Product {asin} - EDA Charts", fontsize=14, fontweight="bold")

    # ---- ROW 0, LEFT: Rating Distribution (Bar Chart) ----
    counts = product_df["rating"].value_counts().sort_index()

    axes[0][0].bar(counts.index, counts.values, color="steelblue")
    axes[0][0].set_xlabel("Star Rating")
    axes[0][0].set_ylabel("Number of Reviews")
    axes[0][0].set_title("Rating Distribution")
    axes[0][0].set_xticks([1, 2, 3, 4, 5])

    # ---- ROW 0, RIGHT: Reviews Over Time (Line Chart) ----
    yearly = product_df.groupby("year").size().reset_index(name="count")

    axes[0][1].plot(yearly["year"], yearly["count"], marker="o", color="coral")
    axes[0][1].set_xlabel("Year")
    axes[0][1].set_ylabel("Number of Reviews")
    axes[0][1].set_title("Reviews Over Time")

    # ---- ROW 1, LEFT: Average Rating Per Year (Line Chart) ----
    yearly_avg = (
        product_df.groupby("year")["rating"]
        .mean()
        .reset_index(name="avg_rating")
    )

    axes[1][0].plot(
        yearly_avg["year"],
        yearly_avg["avg_rating"],
        marker="o",
        color="seagreen",
    )
    axes[1][0].set_xlabel("Year")
    axes[1][0].set_ylabel("Average Star Rating")
    axes[1][0].set_title("Average Rating Over Time")

    # ---- ROW 1, RIGHT: Helpful Votes vs Rating (Bubble Chart) ----
    grouped = (
        product_df.groupby("rating")
        .agg(
            helpful_votes=("helpful_votes", "sum"),
            review_count=("rating", "count"),
        )
        .reset_index()
    )

    axes[1][1].scatter(
        grouped["rating"],
        grouped["helpful_votes"],
        s=grouped["review_count"] * 50,
        alpha=0.6,
        color="mediumpurple",
        edgecolors="black",
    )
    axes[1][1].set_xlabel("Star Rating")
    axes[1][1].set_ylabel("Total Helpful Votes")
    axes[1][1].set_title("Helpful Votes vs Rating (Bubble)")
    axes[1][1].set_xticks([1, 2, 3, 4, 5])

    # ---- ROW 2, LEFT: Top 15 Words (Horizontal Bar Chart) ----
    if top_15:
        top_words = [pair[0] for pair in top_15]
        top_counts = [pair[1] for pair in top_15]

        # Reverse so the most common word is at the top
        top_words.reverse()
        top_counts.reverse()

        axes[2][0].barh(top_words, top_counts, color="darkorange")
        axes[2][0].set_xlabel("Frequency")
        axes[2][0].set_title("Top 15 Words (after NLP cleaning)")

    # ---- ROW 2, RIGHT: Word Cloud ----
    # A word cloud makes the most common words appear BIGGER
    if words:
        clean_text = " ".join(words)

        wc = WordCloud(
            width=400,
            height=300,
            background_color="white",
            colormap="viridis",
        ).generate(clean_text)

        axes[2][1].imshow(wc, interpolation="bilinear")
        axes[2][1].axis("off")
        axes[2][1].set_title("Word Cloud")

    # Adjust spacing so titles and labels don't overlap
    plt.tight_layout()

    plt.show()


# ==========================================================================
# FUNCTION: get_words_by_rating
# Extracts the most common words from reviews at a specific rating range.
# Used to find themes in negative vs positive reviews.
# ==========================================================================
def get_words_by_rating(product_df, min_rating, max_rating, top_n=10):
    """
    Filters reviews to a rating range, cleans the text with NLP,
    and returns the top_n most common words as (word, count) pairs.
    """
    # Filter to reviews in the given rating range
    filtered = product_df[
        product_df["rating"].between(min_rating, max_rating)
    ]

    # Combine all review text, lowercase, remove punctuation
    text = " ".join(filtered["review_text"].tolist()).lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove stop words and lemmatise
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]

    return Counter(words).most_common(top_n)


# ==========================================================================
# FUNCTION: analyse_helpful_votes
# Compares reviews that received helpful votes vs those that didn't.
# Prints insights about what makes a review helpful.
# ==========================================================================
def analyse_helpful_votes(product_df):
    """
    Splits reviews into helpful (got votes) and not helpful (zero votes).
    Compares their average length and rating to find patterns.
    """
    # Add a column with the word count of each review
    product_df["review_length"] = product_df["review_text"].str.split().str.len()

    helpful = product_df[product_df["helpful_votes"] > 0]
    not_helpful = product_df[product_df["helpful_votes"] == 0]

    avg_len_h = helpful["review_length"].mean() if len(helpful) > 0 else 0
    avg_len_n = not_helpful["review_length"].mean() if len(not_helpful) > 0 else 0
    avg_rat_h = helpful["rating"].mean() if len(helpful) > 0 else 0
    avg_rat_n = not_helpful["rating"].mean() if len(not_helpful) > 0 else 0

    print("\n--- WHAT MAKES A REVIEW HELPFUL? ---")
    print(f"  Reviews WITH helpful votes : {len(helpful)}")
    print(f"  Reviews WITHOUT helpful votes: {len(not_helpful)}")
    print(f"  Avg word count (helpful)   : {avg_len_h:.1f}")
    print(f"  Avg word count (not helpful): {avg_len_n:.1f}")
    print(f"  Avg rating (helpful)       : {avg_rat_h:.2f}")
    print(f"  Avg rating (not helpful)   : {avg_rat_n:.2f}")

    if avg_len_h > avg_len_n:
        print("  -> Helpful reviews tend to be LONGER (more detailed).")
    if avg_rat_h < avg_rat_n:
        print("  -> Helpful reviews tend to have LOWER ratings (critical reviews).")


# ==========================================================================
# FUNCTION: show_deeper_charts
# Second chart window with seasonal analysis and theme comparison.
# ==========================================================================
def show_deeper_charts(product_df, asin, low_top, high_top):
    """
    Draws a 2x2 grid of deeper-analysis charts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Product {asin} - Deeper Insights",
        fontsize=14,
        fontweight="bold",
    )

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # ---- TOP-LEFT: Reviews Per Month ----
    month_counts = product_df.groupby("month").size()
    month_counts = month_counts.reindex(range(1, 13), fill_value=0)

    axes[0][0].bar(range(1, 13), month_counts.values, color="steelblue")
    axes[0][0].set_xlabel("Month")
    axes[0][0].set_ylabel("Number of Reviews")
    axes[0][0].set_title("Reviews Per Month (seasonal pattern)")
    axes[0][0].set_xticks(range(1, 13))
    axes[0][0].set_xticklabels(month_labels, rotation=45)

    # ---- TOP-RIGHT: Average Rating Per Month ----
    month_avg = product_df.groupby("month")["rating"].mean()
    month_avg = month_avg.reindex(range(1, 13))

    axes[0][1].plot(range(1, 13), month_avg.values, marker="o", color="seagreen")
    axes[0][1].set_xlabel("Month")
    axes[0][1].set_ylabel("Average Rating")
    axes[0][1].set_title("Average Rating Per Month (seasonal effect)")
    axes[0][1].set_xticks(range(1, 13))
    axes[0][1].set_xticklabels(month_labels, rotation=45)

    # ---- BOTTOM-LEFT: Top Words in Low (1-2 star) Reviews ----
    if low_top:
        low_w = [p[0] for p in low_top[:8]]
        low_c = [p[1] for p in low_top[:8]]
        axes[1][0].barh(low_w[::-1], low_c[::-1], color="salmon")
        axes[1][0].set_xlabel("Frequency")
        axes[1][0].set_title("Top Words in LOW (1-2★) Reviews")
    else:
        axes[1][0].set_title("Not enough low-rating reviews")

    # ---- BOTTOM-RIGHT: Top Words in High (4-5 star) Reviews ----
    if high_top:
        high_w = [p[0] for p in high_top[:8]]
        high_c = [p[1] for p in high_top[:8]]
        axes[1][1].barh(high_w[::-1], high_c[::-1], color="mediumseagreen")
        axes[1][1].set_xlabel("Frequency")
        axes[1][1].set_title("Top Words in HIGH (4-5★) Reviews")
    else:
        axes[1][1].set_title("Not enough high-rating reviews")

    plt.tight_layout()

    plt.show()


# ==========================================================================
# FUNCTION: analyse_product
# Prints summary stats, runs NLP, shows EDA charts, then deeper insights.
# ==========================================================================
def analyse_product(df, asin):
    """
    For a single product:
      1. Filters the DataFrame to only that product's reviews.
      2. Prints summary statistics.
      3. Runs NLP text processing (stop-word removal + lemmatisation).
      4. Shows EDA charts (close window to see next).
      5. Analyses what makes reviews helpful.
      6. Compares themes in low vs high rated reviews.
      7. Shows deeper-insight charts (close window to see next).
    """
    print(f"\n{'=' * 50}")
    print(f"ANALYSIS FOR PRODUCT: {asin}")
    print(f"{'=' * 50}")

    # Filter to just this product
    product_df = df[df["asin"] == asin].copy()

    # ---- Summary statistics ----
    total_reviews = len(product_df)
    average_rating = product_df["rating"].mean()
    median_rating = product_df["rating"].median()
    total_helpful = product_df["helpful_votes"].sum()

    print(f"Total reviews      : {total_reviews}")
    print(f"Average rating     : {average_rating:.2f} / 5")
    print(f"Median rating      : {median_rating}")
    print(f"Total helpful votes: {total_helpful}")

    # Print the rating distribution
    print("\nRating distribution:")
    print(product_df["rating"].value_counts().sort_index())

    # ---- NLP text processing ----
    words, top_15 = process_review_text(product_df)

    print("\nTop 15 words (after stop-word removal & lemmatisation):")
    for word, count in top_15:
        print(f"  {word:20s} : {count}")

    # ---- EDA charts (close the window to see the next set) ----
    show_all_charts(product_df, asin, words, top_15)

    # ---- Helpful-vote analysis (printed to console) ----
    analyse_helpful_votes(product_df)

    # ---- Theme comparison: low vs high rating words ----
    low_top = get_words_by_rating(product_df, 1, 2, top_n=10)
    high_top = get_words_by_rating(product_df, 4, 5, top_n=10)

    print("\n--- COMMON THEMES BY RATING ---")
    print("  Top words in LOW (1-2 star) reviews (issues / complaints):")
    for word, count in low_top:
        print(f"    {word:20s} : {count}")
    print("  Top words in HIGH (4-5 star) reviews (praise / positives):")
    for word, count in high_top:
        print(f"    {word:20s} : {count}")

    # ---- Deeper insight charts (close the window to see the next product) ----
    show_deeper_charts(product_df, asin, low_top, high_top)


# ==========================================================================
# FUNCTION: main
# This is the entry point.  It calls all the other functions in order.
# Without main(), nothing happens when you press Run in IDLE.
# ==========================================================================
def main():
    """
    Runs the full EDA pipeline:
      load -> clean -> check -> pick products -> analyse -> save CSV.
    """
    filepath = "Industrial_and_Scientific_170.json.gz"

    print("=" * 50)
    print("Amazon Reviews EDA")
    print("=" * 50)

    # Step 1 + 2: Load and get a raw DataFrame (pd.read_json does both)
    raw_df = load_reviews(filepath)

    # If loading failed, raw_df is empty -- stop before cleaning crashes
    if raw_df.empty:
        print("No data loaded. Stopping.")
        return

    # Step 3: Clean the data using pandas methods
    df = clean_reviews(raw_df)

    # Step 4: Run data quality checks
    if not check_quality(df):
        print("Data quality checks failed. Stopping.")
        return

    # Step 5: Pick the top products with enough reviews
    selected = pick_products(df, min_reviews=30, how_many=3)

    if not selected:
        print("No products found with enough reviews. Try lowering min_reviews.")
        return

    # Step 6: Analyse each selected product (close each chart window to continue)
    for asin in selected:
        analyse_product(df, asin)

    print("\nAnalysis complete.")


# This "if" guard means: only run main() when you execute the file directly.
# If someone imports this file, the functions are available but main() does
# not run automatically.
if __name__ == "__main__":
    main()
