
import os
import pickle
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "House_prediction_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Cleaned_data.csv"
)


# ============================================================
# CURRENCY
# ============================================================

# Model was trained using the original INR prices.
# The dashboard converts displayed prices from INR to PKR.

INR_TO_PKR = 2.88162

CURRENCY_SYMBOL = "₨"
CURRENCY_CODE = "PKR"


def inr_lakh_to_pkr_lakh(value):
    """
    Convert price from INR Lakhs to PKR Lakhs.
    """
    return float(value) * INR_TO_PKR


def format_pkr_lakh(value):
    """
    Format a price as PKR Lakhs.
    """
    return f"₨ {value:,.2f} L"


def format_pkr(value):
    """
    Format a normal PKR amount.
    """
    return f"₨ {value:,.0f}"


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(" Model file not found.")

    st.write(
        "Expected file:"
    )

    st.code(
        "models/House_prediction_model.pkl"
    )

    st.stop()


try:

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

except Exception as e:

    st.error(" Could not load the model.")
    st.code(str(e))
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):

    st.error(" Cleaned dataset not found.")

    st.write(
        "Expected file:"
    )

    st.code(
        "data/Cleaned_data.csv"
    )

    st.stop()


try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(" Could not load the dataset.")
    st.code(str(e))
    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "location",
    "total_sqft",
    "bath",
    "balcony",
    "bedrooms",
    "price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Missing columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["location"] = (
    df["location"]
    .astype(str)
    .str.strip()
)

locations = sorted(
    df["location"]
    .dropna()
    .unique()
    .tolist()
)


# Original model/data thresholds in INR Lakhs
low_limit_inr = df["price"].quantile(0.33)
high_limit_inr = df["price"].quantile(0.67)

# Display thresholds in PKR Lakhs
low_limit_pkr = inr_lakh_to_pkr_lakh(
    low_limit_inr
)

high_limit_pkr = inr_lakh_to_pkr_lakh(
    high_limit_inr
)


def get_price_category(price_inr):
    """
    Categorize the original model prediction
    using the original INR dataset thresholds.
    """

    if price_inr <= low_limit_inr:
        return "LOW PRICE"

    elif price_inr <= high_limit_inr:
        return "MEDIUM PRICE"

    else:
        return "HIGH PRICE"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("House Price Prediction")

    st.caption(
        "Machine Learning Dashboard"
    )

    st.divider()

    st.subheader("Dashboard Menu")

    page = st.radio(
        "Select Section",
        [
            "Home",
            "Property Prediction",
            "Price Analysis",
            "Model Information",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Price Categories")

    st.write(
        f"LOW: {format_pkr_lakh(low_limit_pkr)} or below"
    )

    st.write(
        f"MEDIUM: {format_pkr_lakh(low_limit_pkr)} - "
        f"{format_pkr_lakh(high_limit_pkr)}"
    )

    st.write(
        f"HIGH: Above {format_pkr_lakh(high_limit_pkr)}"
    )

    st.divider()

    st.subheader("Dataset Information")

    st.write(
        f"Properties: {len(df):,}"
    )

    st.write(
        f"Locations: {df['location'].nunique():,}"
    )

    st.write(
        "Currency: PKR"
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title(
        "House Price Prediction"
    )

    st.caption(
        "Machine Learning Based Property Price Estimator"
    )

    st.divider()

    # --------------------------------------------------------
    # Dashboard Overview
    # --------------------------------------------------------

    st.header(" Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "Property Prediction"
        )

        st.write(
            "Enter the property location, area, "
            "bedrooms, bathrooms and balconies "
            "to estimate the expected house price."
        )

    with col2:

        st.subheader(
            "Price Analysis"
        )

        st.write(
            "Explore property prices, area "
            "distribution and important statistics "
            "from the cleaned dataset."
        )

    with col3:

        st.subheader(
            "Machine Learning"
        )

        st.write(
            "A trained Linear Regression pipeline "
            "processes property information and "
            "produces an estimated house price."
        )

    st.divider()

    # --------------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------------

    st.header(" Dataset Overview")

    average_price_pkr = inr_lakh_to_pkr_lakh(
        df["price"].mean()
    )

    max_price_pkr = inr_lakh_to_pkr_lakh(
        df["price"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            " Properties",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Locations",
            f"{df['location'].nunique():,}"
        )

    with col3:

        st.metric(
            "Average Price",
            format_pkr_lakh(
                average_price_pkr
            )
        )

    with col4:

        st.metric(
            "Average Area",
            f"{df['total_sqft'].mean():,.0f} sqft"
        )

    st.divider()

    # --------------------------------------------------------
    # How to Use
    # --------------------------------------------------------

    st.header("How to Use")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("1. Enter Details")

        st.write(
            "Open Property Prediction and "
            "enter the property information."
        )

    with col2:

        st.subheader("2. Get Prediction")

        st.write(
            "Click the prediction button to "
            "generate the estimated house price."
        )

    with col3:

        st.subheader("3. Analyze")

        st.write(
            "Use Price Analysis to explore "
            "the dataset and price distribution."
        )

    st.divider()

    # --------------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------------

    st.header(" Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PROPERTY PREDICTION
# ============================================================

elif page == "Property Prediction":

    st.title("Property Price Prediction")

    st.caption(
        "Enter property details to predict the house price using the trained model."
    )

    st.divider()

    st.header("Property Details")

    col1, col2 = st.columns(2)

    with col1:

        default_location = (
            "Electronic City Phase II"
            if "Electronic City Phase II" in locations
            else locations[0]
        )

        location = st.selectbox(
            "Location",
            locations,
            index=locations.index(default_location)
        )

        total_sqft = st.number_input(
            "Total Area (Sqft)",
            min_value=300.0,
            max_value=10000.0,
            value=1200.0,
            step=50.0
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

    with col2:

        bath = st.number_input(
            "Bathrooms",
            min_value=1.0,
            max_value=20.0,
            value=2.0,
            step=1.0
        )

        balcony = st.number_input(
            "Balconies",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=1.0
        )

    st.divider()

    predict = st.button(
        "Predict House Price",
        use_container_width=True
    )

    if predict:

        input_data = pd.DataFrame(
            [[
                location,
                total_sqft,
                bath,
                balcony,
                bedrooms
            ]],
            columns=[
                "location",
                "total_sqft",
                "bath",
                "balcony",
                "bedrooms"
            ]
        )

        try:

            # The trained model predicts the original INR price in Lakhs.
            prediction_inr = float(
                model.predict(input_data)[0]
            )

            # Categorize using the original INR dataset thresholds.
            category = get_price_category(
                prediction_inr
            )

            # Calculate INR price per square foot.
            price_per_sqft_inr = (
                prediction_inr * 100000
            ) / total_sqft

            st.divider()

            st.header("Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Estimated Price",
                    f"₹ {prediction_inr:,.2f} L"
                )

            with col2:

                st.metric(
                    "Price per Sqft",
                    f"₹ {price_per_sqft_inr:,.0f}"
                )

            with col3:

                st.metric(
                    "Price Category",
                    category
                )

            if category == "LOW PRICE":

                st.success(
                    "This property falls into the LOW PRICE category."
                )

            elif category == "MEDIUM PRICE":

                st.info(
                    "This property falls into the MEDIUM PRICE category."
                )

            else:

                st.warning(
                    "This property falls into the HIGH PRICE category."
                )

            st.caption(
                "Prediction is generated directly by the trained model "
                "using the original INR price scale."
            )

            st.divider()

            st.header("Property Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Location",
                    location
                )

            with col2:

                st.metric(
                    "Area",
                    f"{total_sqft:,.0f} sqft"
                )

            with col3:

                st.metric(
                    "Bedrooms",
                    bedrooms
                )

            with col4:

                st.metric(
                    "Bathrooms",
                    bath
                )

            st.divider()

            st.header("Price Comparison")

            low_price = float(low_limit_inr)

            medium_price = float(
                (low_limit_inr + high_limit_inr) / 2
            )

            high_price = float(high_limit_inr)

            categories = [
                "Low Price",
                "Medium Price",
                "Your Prediction",
                "High Price"
            ]

            values = [
                low_price,
                medium_price,
                prediction_inr,
                high_price
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    text=[
                        f"₹ {value:,.2f} L"
                        for value in values
                    ],
                    textposition="outside",
                    marker_color=[
                        "#3FB950",
                        "#FFD700",
                        "#00BFFF",
                        "#FF4B4B"
                    ]
                )
            )

            fig.update_layout(
                title="House Price Comparison",
                xaxis_title="Category",
                yaxis_title="Price (INR Lakhs)",
                height=500,
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.code(str(e))


# ============================================================
# PRICE ANALYSIS
# ============================================================

elif page == "Price Analysis":

    st.title(
        "Price Analysis"
    )

    st.caption(
        "Explore house prices and property characteristics."
    )

    st.divider()

    # --------------------------------------------------------
    # Price Statistics
    # --------------------------------------------------------

    st.header("Price Statistics")

    average_price = inr_lakh_to_pkr_lakh(
        df["price"].mean()
    )

    median_price = inr_lakh_to_pkr_lakh(
        df["price"].median()
    )

    minimum_price = inr_lakh_to_pkr_lakh(
        df["price"].min()
    )

    maximum_price = inr_lakh_to_pkr_lakh(
        df["price"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Average",
            format_pkr_lakh(
                average_price
            )
        )

    with col2:

        st.metric(
            "Median",
            format_pkr_lakh(
                median_price
            )
        )

    with col3:

        st.metric(
            "Minimum",
            format_pkr_lakh(
                minimum_price
            )
        )

    with col4:

        st.metric(
            "Maximum",
            format_pkr_lakh(
                maximum_price
            )
        )

    st.divider()

    # --------------------------------------------------------
    # Price Distribution
    # --------------------------------------------------------

    st.header("House Price Distribution")

    price_pkr_lakh = (
        df["price"]
        * INR_TO_PKR
    )

    fig_price = go.Figure()

    fig_price.add_trace(
        go.Histogram(
            x=price_pkr_lakh,
            nbinsx=50,
            marker_color="#D4AF37"
        )
    )

    fig_price.update_layout(
        title="Distribution of House Prices",
        xaxis_title="Price (PKR Lakhs)",
        yaxis_title="Number of Properties",
        height=500,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # Area Distribution
    # --------------------------------------------------------

    st.header("Property Area Distribution")

    fig_area = go.Figure()

    fig_area.add_trace(
        go.Histogram(
            x=df["total_sqft"],
            nbinsx=50,
            marker_color="#00BFFF"
        )
    )

    fig_area.update_layout(
        title="Distribution of Property Areas",
        xaxis_title="Area (Sqft)",
        yaxis_title="Number of Properties",
        height=500,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_area,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # Dataset Statistics
    # --------------------------------------------------------

    st.header("Dataset Statistics")

    statistics = df[
        [
            "total_sqft",
            "bath",
            "balcony",
            "bedrooms"
        ]
    ].describe()

    statistics = statistics.round(2)

    st.dataframe(
        statistics,
        use_container_width=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.title(
        "Model Information"
    )

    st.caption(
        "Technical details of the Machine Learning model."
    )

    st.divider()

    st.header(" Machine Learning Model")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Algorithm")

        st.write(
            "Linear Regression"
        )

        st.subheader("Problem Type")

        st.write(
            "Regression"
        )

        st.subheader("Target Variable")

        st.write(
            "House Price"
        )

    with col2:

        st.subheader("Preprocessing")

        st.write(
            "One-Hot Encoding for location"
        )

        st.write(
            "Standard Scaling"
        )

        st.write(
            "Pipeline-based preprocessing"
        )

    st.divider()

    st.header(" Model Input Features")

    feature_df = pd.DataFrame(
        {
            "Feature": [
                "location",
                "total_sqft",
                "bath",
                "balcony",
                "bedrooms"
            ],
            "Meaning": [
                "Property location",
                "Total property area",
                "Number of bathrooms",
                "Number of balconies",
                "Number of bedrooms"
            ]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.header(" Training Workflow")

    workflow = [
        "1. Load the property dataset",
        "2. Handle missing values",
        "3. Remove duplicate records",
        "4. Clean property locations",
        "5. Convert property size into bedrooms",
        "6. Clean total square feet values",
        "7. Remove unrealistic property values",
        "8. Apply feature preprocessing",
        "9. Train the Linear Regression model",
        "10. Save the trained model"
    ]

    for step in workflow:

        st.write(step)

    st.divider()

    st.info(
        "The deployed application uses the same feature structure "
        "that was used during model training."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title(
        "About House Price Prediction"
    )

    st.caption(
        "A complete Machine Learning project for property price estimation"
    )

    st.divider()

    # --------------------------------------------------------
    # Project Introduction
    # --------------------------------------------------------

    st.header("Project Introduction")

    st.write(
        """
        House Price Prediction is an end-to-end Machine Learning
        project designed to estimate residential property prices
        using important property characteristics such as location,
        total area, bedrooms, bathrooms and balconies.
        """
    )

    st.write(
        """
        The project demonstrates the complete Machine Learning
        workflow — from data preparation and feature engineering
        to model training, prediction and deployment through an
        interactive Streamlit dashboard.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Problem Statement
    # --------------------------------------------------------

    st.header("Problem Statement")

    st.write(
        """
        Property prices can vary significantly based on location,
        property size and the number of rooms and facilities.

        Manually estimating a property's value can therefore be
        difficult and inconsistent. This project uses historical
        property data to build a Machine Learning model that can
        provide an estimated property price from a selected set
        of features.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Business Context
    # --------------------------------------------------------

    st.header("Business Context")

    st.write(
        """
        A house price prediction system can provide useful
        preliminary estimates for real-estate related decisions.
        Potential applications include:
        """
    )

    business_uses = [
        "Helping buyers estimate the approximate value of a property.",
        "Helping sellers understand a reasonable price range.",
        "Supporting real-estate businesses with data-driven estimates.",
        "Providing a quick starting point for property evaluation.",
        "Demonstrating how Machine Learning can automate price estimation."
    ]

    for item in business_uses:

        st.write(item)

    st.divider()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Properties",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Locations",
            f"{df['location'].nunique():,}"
        )

    with col3:

        st.metric(
            "Input Features",
            "5"
        )

    with col4:

        st.metric(
            "Target",
            "House Price"
        )

    st.write(
        """
        The model uses five main input features:
        """
    )

    features = pd.DataFrame(
        {
            "Feature": [
                "Location",
                "Total Sqft",
                "Bedrooms",
                "Bathrooms",
                "Balconies"
            ],
            "Purpose": [
                "Represents the property's area/location.",
                "Represents the total property size.",
                "Represents the number of bedrooms.",
                "Represents the number of bathrooms.",
                "Represents the number of balconies."
            ]
        }
    )

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # Machine Learning Approach
    # --------------------------------------------------------

    st.header("Machine Learning Approach")

    st.write(
        """
        The project uses Linear Regression as the primary
        regression algorithm. The categorical location feature
        is converted into numerical representation using
        One-Hot Encoding, while numerical features are scaled
        using StandardScaler.
        """
    )

    st.write(
        """
        These preprocessing steps and the regression model are
        combined into a pipeline so that the same transformations
        are automatically applied when a new property is submitted
        through the dashboard.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Data Preparation
    # --------------------------------------------------------

    st.header("Data Preparation")

    preparation_steps = [
        "Handle missing values.",
        "Remove unnecessary columns.",
        "Remove duplicate records.",
        "Clean and standardize location names.",
        "Group low-frequency locations into an 'other' category.",
        "Extract the number of bedrooms from property-size information.",
        "Convert square-foot ranges into usable numerical values.",
        "Create useful derived measurements for data filtering.",
        "Remove unrealistic property records.",
        "Prepare the final features for Machine Learning."
    ]

    for index, step in enumerate(
        preparation_steps,
        start=1
    ):

        st.write(
            f"{index}. {step}"
        )

    st.divider()

    # --------------------------------------------------------
    # Technologies
    # --------------------------------------------------------

    st.header("Technologies Used")

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "NumPy",
                "Scikit-learn",
                "Plotly",
                "Streamlit",
                "Pickle"
            ],
            "Purpose": [
                "Core programming language",
                "Data cleaning and manipulation",
                "Numerical computation",
                "Machine Learning and preprocessing",
                "Interactive data visualization",
                "Dashboard and deployment",
                "Model serialization"
            ]
        }
    )

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # Project Workflow
    # --------------------------------------------------------

    st.header("End-to-End Project Workflow")

    workflow = [
        ("01", "Data Collection", "Obtain the property dataset."),
        ("02", "Data Understanding", "Understand columns, target and data quality."),
        ("03", "Data Cleaning", "Handle missing values and duplicates."),
        ("04", "Feature Engineering", "Create useful property features."),
        ("05", "Outlier Handling", "Remove unrealistic property records."),
        ("06", "Preprocessing", "Encode categorical data and scale numerical features."),
        ("07", "Model Training", "Train the Linear Regression model."),
        ("08", "Model Evaluation", "Evaluate the trained model."),
        ("09", "Model Saving", "Save the trained pipeline as a PKL file."),
        ("10", "Deployment", "Deploy the model through Streamlit.")
    ]

    for number, title, description in workflow:

        col1, col2 = st.columns([1, 4])

        with col1:

            st.subheader(number)

        with col2:

            st.write(
                f"**{title}** — {description}"
            )

    st.divider()

    # --------------------------------------------------------
    # Dashboard Features
    # --------------------------------------------------------

    st.header("Dashboard Features")

    dashboard_features = [
        "Clean and simple home dashboard.",
        " Interactive property price prediction.",
        " Price and area distribution analysis.",
        " Dataset statistics.",
        " Machine Learning model information.",
        " PKR-based price display.",
        " Responsive Streamlit interface.",
        " Real-time prediction from the trained model."
    ]

    for feature in dashboard_features:

        st.write(feature)

    st.divider()

    # --------------------------------------------------------
    # Currency Information
    # --------------------------------------------------------

    st.header("Currency Conversion")

    st.write(
        f"""
        The original Bengaluru dataset contains prices in Indian
        Rupees (INR). The Machine Learning model therefore remains
        trained on the original INR values.

        For dashboard presentation, predicted and displayed prices
        are converted to Pakistani Rupees (PKR) using the configured
        INR-to-PKR exchange rate.
        """
    )

    st.info(
        f"Configured conversion: 1 INR ≈ {INR_TO_PKR:.5f} PKR"
    )

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    st.divider()

    st.header("Project Limitations")

    limitations = [
        "The prediction is an estimate and not a guaranteed market price.",
        "Actual property prices may depend on factors not included in the dataset.",
        "Market conditions can change over time.",
        "Currency conversion rates fluctuate and can change the displayed PKR value.",
        "The model's prediction quality depends on the quality and coverage of the training data."
    ]

    for limitation in limitations:

        st.write(
            f"• {limitation}"
        )

    st.divider()

    # --------------------------------------------------------
    # Final Project Summary
    # --------------------------------------------------------

    st.header("Project Summary")

    st.write(
        """
        House Price Prediction demonstrates how a complete
        Machine Learning solution can move from raw data to a
        usable application.

        The project combines data cleaning, feature engineering,
        preprocessing, regression modeling, visualization and
        Streamlit deployment into one practical portfolio project.
        """
    )

    st.success(
        "House Price Prediction — End-to-End Machine Learning Project"
    )
