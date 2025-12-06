import os
import datetime as dt
import mlflow
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_URI = os.path.join(BASE_DIR, "xgb_price_model_best")

DATA_PATH = os.path.join(BASE_DIR, "final_predictions_clean_with_corrected_mileage.csv")


DATA_PATH = "final_predictions_clean_with_corrected_mileage.csv"

FEATURE_COLS = [
    "year",
    "mileage",
    "mileage_corrected",
    "mileage_numeric",
    "mileage_from_title",
    "mileage_from_details",
    "views",
    "watchers",
    "comments",
    "accidents",
    "latitude",
    "longitude",
    "auction_month",
    "auction_dow",
]

BEST_MONTH = 3
BEST_MONTH_LABEL = "February to April"

BEST_DOW = 4
BEST_DOW_LABEL = "Thursday or Friday"


# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)


@st.cache_data
def load_training_stats(csv_path):
    df = pd.read_csv(csv_path)

    med_lat = df["latitude"].median()
    med_lon = df["longitude"].median()
    med_views = df["views"].median() if "views" in df.columns else 8000
    med_watchers = df["watchers"].median() if "watchers" in df.columns else 120
    med_comments = df["comments"].median() if "comments" in df.columns else 25

    if "accidents" in df.columns and is_numeric_dtype(df["accidents"]):
        med_accidents = int(round(df["accidents"].median()))
    else:
        med_accidents = 0

    return med_lat, med_lon, med_views, med_watchers, med_comments, med_accidents


# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------

def set_page_style():
    st.set_page_config(
        page_title="Porsche 911 Auction Intelligence",
        page_icon="🚗",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            body { background-color: #000; color: #fff; }
            .main { background-color: #000; }
            h1, h2, h3, h4 { color: #f7941d; }
            .avant-card {
                background-color: #111;
                border-radius: 12px;
                padding: 1.2rem 1.4rem;
                border: 1px solid #333;
            }
            .avant-metric {
                font-size: 1.8rem;
                font-weight: 800;
                color: #f7941d;
            }
            .avant-subtitle { color: #ccc; font-size: 0.95rem; }
            .stButton>button {
                background-color: #f7941d; color: black;
                border-radius: 999px; padding: 0.5rem 1.4rem;
                font-weight: 600;
            }
            .stButton>button:hover {
                background-color: #ffae42;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# FEATURE BUILDER
# ------------------------------------------------------------------

def build_feature_row(year, mileage, views, watchers, comments, accidents, lat, lon, auction_month, auction_dow):
    row = {
        "year": year,
        "mileage": mileage,
        "mileage_corrected": mileage,
        "mileage_numeric": mileage,
        "mileage_from_title": mileage,
        "mileage_from_details": mileage,
        "views": views,
        "watchers": watchers,
        "comments": comments,
        "accidents": accidents,
        "latitude": lat,
        "longitude": lon,
        "auction_month": auction_month,
        "auction_dow": auction_dow,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------

def main():
    set_page_style()

    model = load_model()
    med_lat, med_lon, med_views, med_watchers, med_comments, med_accidents = load_training_stats(DATA_PATH)

    # ---------------- HERO BANNER (Left: Logo/Title, Right: Image) ----------------
    
    # ADJUSTED RATIO: Increased the left column from 1 to 1.2 for a larger logo area
    banner_left_col, banner_right_col = st.columns([1.2, 4]) 

    # --- LEFT BANNER CONTENT (Logo and Title) ---
    with banner_left_col:
        # ADJUSTED WIDTH: Increased the logo width from 300px to 350px
        st.image("avant_garde_logo.png", width=350) 
        
        st.markdown("<h1>Porsche 911 Auction Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<div class='avant-subtitle'>Price advisor for Bring a Trailer–style auctions</div>", unsafe_allow_html=True)
        st.markdown("<div class='avant-subtitle'>Mark Barlow • MS AIB Candidate</div>", unsafe_allow_html=True)

    # --- RIGHT BANNER CONTENT (Hero Image) ---
    with banner_right_col:
        # Assuming your hero image is named 'GT3.jpg' based on the file uploads
        st.image('GT3.jpg', use_column_width=True) 

    st.markdown("---")

    # ---------------- LAYOUT ----------------
    input_col, explain_col = st.columns([1.4, 1])

    # ---------------- INPUTS ----------------
    with input_col:
        st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
        st.markdown("#### Configure the car")

        years = list(range(1965, 2024))
        year = st.selectbox("Model Year", years, index=years.index(2015))

        mileage = st.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=250000,
            value=30000,
            step=500
        )

        submodel = st.selectbox(
            "Submodel",
            ["Base", "Carrera", "Carrera S", "Carrera 4S", "Targa",
             "Turbo", "Turbo S", "GT3", "GT3 RS", "GT2 RS", "Other"],
            index=7,
        )

        title = st.text_input("Auction Title", "2016 Porsche 911 GT3 RS - PCCB, Lift, 1-Owner")
        zipcode = st.text_input("Seller ZIP Code", "85260")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------- EXPLANATION ---------------
    with explain_col:
        st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
        st.subheader("Why these features drive price")
        st.markdown(
        """
    - **Submodel** is the single largest driver of Porsche 911 prices because it captures performance level, trim hierarchy, rarity, and enthusiast demand.
    - **Year** reflects generational improvements, technology updates, and market scarcity, especially for older air-cooled or special-edition models.
    - **Mileage** is the strongest proxy for condition, with low-mileage cars commanding significant premiums.
    - **Owners** provides insight into vehicle history. Fewer owners often signals better care, stronger provenance, and higher perceived value.
    - **Title text** helps capture important descriptors such as special options, unique builds, documentation history, and keywords that influence buyer behavior.
    - **Location (ZIP code)** maps to regional demand, shipping considerations, and bidding intensity differences across U.S. markets.
        """
    )
        st.markdown("</div>", unsafe_allow_html=True)

    # divider before next section
        st.markdown("---")
        st.markdown("#### Recommended auction timing")


    # ---------------- PRICE ESTIMATE ----------------
    st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
    st.markdown("### Price Estimate")

    if st.button("Estimate Auction Price"):
        features = build_feature_row(
            year, mileage,
            med_views, med_watchers, med_comments,
            med_accidents, med_lat, med_lon,
            BEST_MONTH, BEST_DOW
        )

        predicted_price = float(model.predict(features)[0])

        c1, c2, c3 = st.columns([1.3, 1, 1])

        with c1:
            st.markdown("Predicted Sale Price")
            st.markdown(f"<div class='avant-metric'>${predicted_price:,.0f}</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("Recommended Window")
            st.markdown(f"<div class='avant-metric'>{BEST_MONTH_LABEL}</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("Recommended End Day")
            st.markdown(f"<div class='avant-metric'>{BEST_DOW_LABEL}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f"""
            This estimate assumes a **{year} {submodel}** with **{mileage:,} miles**,  
            listed from ZIP **{zipcode}** and marketed in a typical BaT-style format.
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------- FOOTER (W.P. Carey Logo) ----------------
    st.markdown("<br><br><br>", unsafe_allow_html=True) # Add some vertical space
    st.markdown("---") 

    footer_col1, footer_col2, footer_col3 = st.columns([1, 4, 1])
    
    with footer_col2:
        # Place the W.P. Carey logo centrally using the middle column
        st.image(
            "asu-wpcarey-school-of-business-asu-footer.png", 
            width=350,  # Adjust width as needed for your footer style
            caption="Powered by ASU W.P. Carey School of Business Research"
        )
    
    # Ensure the main content doesn't run off the bottom
    st.markdown("<br><br>", unsafe_allow_html=True) 


if __name__ == "__main__":
    main()
