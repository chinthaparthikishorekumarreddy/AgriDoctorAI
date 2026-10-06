import streamlit as st
import torch
import requests
import io

from PIL import Image
from transformers import AutoModelForImageClassification
from torchvision import transforms
from gtts import gTTS


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgriDoctor AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0b6b3a, #198754);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}

.feature-card {
    padding: 18px;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    text-align: center;
    min-height: 150px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background: #ffffff;
    border-left: 6px solid #198754;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🌱 AgriDoctor AI</h1>
    <p>AI-Powered Crop Doctor & Smart Agricultural Advisory</p>
    <p>
        Detect possible crop diseases, understand farm conditions,
        and receive weather-based irrigation guidance.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌱 AgriDoctor AI")

    st.markdown("### Farmer Assistant")

    page = st.radio(
        "Choose a feature",
        [
            "🏠 Dashboard",
            "🔬 AI Crop Doctor",
            "💧 Smart Irrigation"
        ]
    )

    st.markdown("---")

    st.markdown("### 🚀 System Features")

    st.write("🤖 AI Disease Screening")
    st.write("📊 Confidence Analysis")
    st.write("🌦️ Live Weather")
    st.write("💧 Irrigation Advisory")
    st.write("🇮🇳 Telugu Support")
    st.write("🔊 Voice Advisory")

    st.markdown("---")

    st.caption(
        "SGDC Hackthon 2K26\n\n"
        "AET-03: Smart Agriculture"
    )


# =========================================================
# AI MODEL
# =========================================================

MODEL_NAME = (
    "linkanjarad/"
    "mobilenet_v2_1.0_224-plant-disease-identification"
)


@st.cache_resource
def load_model():

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return model


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================================================
# CROP DISEASE ADVISORY
# =========================================================

def disease_advice(disease):

    disease_lower = disease.lower()

    if "healthy" in disease_lower:

        return {
            "risk": "Low",
            "action": "Continue normal crop monitoring.",
            "prevention": (
                "Maintain proper irrigation, nutrition, "
                "airflow and regular crop inspection."
            )
        }

    if "blight" in disease_lower:

        return {
            "risk": "High",
            "action": (
                "Inspect affected leaves and remove severely "
                "damaged plant material where appropriate."
            ),
            "prevention": (
                "Avoid excessive moisture and unnecessary "
                "overhead irrigation."
            )
        }

    if "mildew" in disease_lower:

        return {
            "risk": "Moderate",
            "action": (
                "Improve airflow and avoid keeping leaves "
                "wet for long periods."
            ),
            "prevention": (
                "Monitor plants regularly and maintain "
                "appropriate spacing."
            )
        }

    if "spot" in disease_lower:

        return {
            "risk": "Moderate",
            "action": (
                "Inspect nearby plants for similar symptoms "
                "and remove severely affected leaves when appropriate."
            ),
            "prevention": (
                "Avoid unnecessary leaf wetness and maintain "
                "good field hygiene."
            )
        }

    if "rot" in disease_lower:

        return {
            "risk": "High",
            "action": (
                "Inspect affected plant parts and improve "
                "drainage and airflow."
            ),
            "prevention": (
                "Avoid excessive irrigation and monitor "
                "moisture conditions."
            )
        }

    return {
        "risk": "Moderate / Uncertain",
        "action": (
            "Inspect the crop carefully and monitor nearby "
            "plants for similar symptoms."
        ),
        "prevention": (
            "Maintain good airflow and avoid unnecessary "
            "overwatering."
        )
    }


# =========================================================
# CROP EXTRACTION
# =========================================================

def get_crop_name(disease):

    crops = [
        "Grape",
        "Tomato",
        "Potato",
        "Peach",
        "Bell Pepper",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry"
    ]

    for crop in crops:

        if crop.lower() in disease.lower():
            return crop

    return "Unknown"


# =========================================================
# WEATHER FUNCTION
# =========================================================

def get_weather(location):

    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geo_params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geo_response = requests.get(
        geocode_url,
        params=geo_params,
        timeout=10
    )

    geo_data = geo_response.json()

    if not geo_data.get("results"):
        return None

    place = geo_data["results"][0]

    latitude = place["latitude"]
    longitude = place["longitude"]

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max"
        ),
        "timezone": "auto",
        "forecast_days": 1
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=10
    )

    data = weather_response.json()

    return {
        "place": (
            f"{place['name']}, "
            f"{place.get('country', '')}"
        ),
        "temperature": data["current"]["temperature_2m"],
        "humidity": data["current"]["relative_humidity_2m"],
        "current_rain": data["current"]["precipitation"],
        "max_temp": data["daily"]["temperature_2m_max"][0],
        "min_temp": data["daily"]["temperature_2m_min"][0],
        "rainfall": data["daily"]["precipitation_sum"][0],
        "rain_probability": (
            data["daily"]["precipitation_probability_max"][0]
        )
    }


# =========================================================
# IRRIGATION LOGIC
# =========================================================

def irrigation_decision(
    soil_moisture,
    rain_probability,
    rainfall
):

    if soil_moisture >= 75:

        return (
            "DO NOT IRRIGATE",
            "The soil is already wet. Additional irrigation "
            "may increase waterlogging and disease risk."
        )

    if rain_probability >= 70:

        return (
            "DELAY IRRIGATION",
            "Rain is highly likely today. Additional irrigation "
            "may not be necessary."
        )

    if soil_moisture <= 30:

        return (
            "IRRIGATE",
            "The soil appears dry. Irrigation may be required, "
            "but check actual root-zone moisture first."
        )

    if soil_moisture <= 50:

        return (
            "LIGHT IRRIGATION",
            "Soil moisture is getting low. Consider moderate "
            "irrigation if rainfall is not expected."
        )

    return (
        "MONITOR",
        "Soil moisture is currently moderate. Continue monitoring."
    )


# =========================================================
# FARMER LANGUAGE
# =========================================================

def farmer_message(
    language,
    crop,
    decision,
    rain_probability
):

    if language == "తెలుగు":

        if decision == "DO NOT IRRIGATE":

            return (
                f"మీ {crop} పంటకు ప్రస్తుతం నీరు పెట్టకుండా "
                f"నేల తేమను గమనించండి. వర్షం వచ్చే అవకాశం "
                f"{rain_probability}% ఉంది. ఎక్కువ నీరు పెడితే "
                f"వేర్లు దెబ్బతినే అవకాశం ఉంది."
            )

        if decision == "DELAY IRRIGATION":

            return (
                f"మీ {crop} పంటకు ప్రస్తుతం నీరు పెట్టడం ఆలస్యం చేయండి. "
                f"ఈరోజు వర్షం వచ్చే అవకాశం {rain_probability}% ఉంది."
            )

        if decision == "IRRIGATE":

            return (
                f"మీ {crop} పంటకు నేల తేమ తక్కువగా ఉంది. "
                f"అవసరమైన మేరకు నీరు పెట్టండి. అధికంగా నీరు పెట్టవద్దు."
            )

        if decision == "LIGHT IRRIGATION":

            return (
                f"మీ {crop} పంటలో నేల తేమ కొంత తక్కువగా ఉంది. "
                f"వాతావరణాన్ని గమనించి అవసరమైతే కొద్దిగా నీరు పెట్టండి."
            )

        return (
            f"మీ {crop} పంటలో నేల తేమ ప్రస్తుతం సాధారణంగా ఉంది. "
            f"కొనసాగిస్తూ గమనించండి."
        )

    else:

        if decision == "DO NOT IRRIGATE":

            return (
                f"Do not irrigate the {crop} crop now. "
                f"The soil is already wet and rain probability "
                f"is {rain_probability} percent."
            )

        if decision == "DELAY IRRIGATION":

            return (
                f"Delay irrigation for the {crop} crop. "
                f"Rain probability is {rain_probability} percent today."
            )

        if decision == "IRRIGATE":

            return (
                f"The {crop} crop may need irrigation because "
                f"soil moisture is low. Irrigate moderately."
            )

        if decision == "LIGHT IRRIGATION":

            return (
                f"Soil moisture for the {crop} crop is getting low. "
                f"Consider light irrigation if rain is not expected."
            )

        return (
            f"Soil moisture for the {crop} crop is currently normal. "
            f"Continue monitoring."
        )


# =========================================================
# VOICE FUNCTION
# =========================================================

def create_voice(text, language):

    try:

        lang_code = "te" if language == "తెలుగు" else "en"

        audio = io.BytesIO()

        tts = gTTS(
            text=text,
            lang=lang_code,
            slow=False
        )

        tts.write_to_fp(audio)

        audio.seek(0)

        return audio

    except Exception:

        return None


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header("🏠 Farmer Dashboard")

    st.write(
        "AgriDoctor AI combines AI crop screening, weather "
        "information, soil moisture and farmer-friendly advice."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="feature-card">
        <h2>🤖</h2>
        <h3>AI Crop Doctor</h3>
        <p>Screen crop images for possible diseases.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="feature-card">
        <h2>🌦️</h2>
        <h3>Weather Intelligence</h3>
        <p>Use live weather information for farm decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="feature-card">
        <h2>💧</h2>
        <h3>Smart Irrigation</h3>
        <p>Combine soil moisture and rain probability.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🇮🇳 Farmer-Centric Features")

    st.write("✅ Telugu farmer advisory")
    st.write("✅ English advisory")
    st.write("✅ Voice recommendations")
    st.write("✅ AI confidence warning")
    st.write("✅ Expert recommendation when uncertain")

    st.info(
        "💡 AgriDoctor AI is designed as an AI screening "
        "and advisory tool. It does not replace professional "
        "agricultural diagnosis."
    )


# =========================================================
# AI CROP DOCTOR
# =========================================================

elif page == "🔬 AI Crop Doctor":

    st.header("🔬 AI Crop Doctor")

    st.write(
        "Upload a clear close-up photograph of one crop leaf."
    )

    uploaded_file = st.file_uploader(
        "📷 Choose crop image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Crop Image",
            width=500
        )

        st.success(
            "✅ Image uploaded successfully!"
        )

        if st.button(
            "🔍 Analyze Crop",
            type="primary"
        ):

            with st.spinner(
                "🤖 AI is analyzing your crop..."
            ):

                try:

                    model = load_model()

                    image_tensor = image_transform(
                        image
                    ).unsqueeze(0)

                    with torch.no_grad():

                        outputs = model(
                            pixel_values=image_tensor
                        )

                        probabilities = torch.softmax(
                            outputs.logits,
                            dim=-1
                        )

                    top_values, top_indices = torch.topk(
                        probabilities,
                        k=3,
                        dim=1
                    )

                    confidence = (
                        top_values[0][0].item() * 100
                    )

                    predicted_class = (
                        top_indices[0][0].item()
                    )

                    disease_name = (
                        model.config.id2label[
                            predicted_class
                        ]
                    )

                    crop = get_crop_name(
                        disease_name
                    )

                    advice = disease_advice(
                        disease_name
                    )

                    # -------------------------------------
                    # RESULT
                    # -------------------------------------

                    st.markdown("---")

                    st.header(
                        "🔬 AI Analysis Result"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Detected Condition",
                            disease_name
                        )

                    with col2:

                        st.metric(
                            "AI Confidence",
                            f"{confidence:.1f}%"
                        )

                    with col3:

                        st.metric(
                            "Risk Level",
                            advice["risk"]
                        )

                    # -------------------------------------
                    # CONFIDENCE WARNING
                    # -------------------------------------

                    if confidence < 60:

                        st.warning(
                            "⚠️ Low AI confidence. "
                            "The image may not match the model's "
                            "known crop/disease categories."
                        )

                        st.info(
                            "📷 For better results, upload a "
                            "clear close-up photo of one affected leaf."
                        )

                    elif confidence < 80:

                        st.warning(
                            "⚠️ Moderate confidence. "
                            "Treat this as an early screening result."
                        )

                    else:

                        st.success(
                            "✅ High AI confidence. "
                            "Still confirm important decisions "
                            "with an agricultural expert."
                        )

                    # -------------------------------------
                    # TOP 3
                    # -------------------------------------

                    st.subheader(
                        "📊 Top 3 AI Predictions"
                    )

                    for i in range(3):

                        class_id = (
                            top_indices[0][i].item()
                        )

                        label = (
                            model.config.id2label[
                                class_id
                            ]
                        )

                        score = (
                            top_values[0][i].item()
                            * 100
                        )

                        st.write(
                            f"**{i + 1}. {label}** — "
                            f"{score:.1f}%"
                        )

                        st.progress(
                            min(
                                top_values[0][i].item(),
                                1.0
                            )
                        )

                    # -------------------------------------
                    # ADVISORY
                    # -------------------------------------

                    st.markdown("---")

                    st.header(
                        "💡 Agricultural Advisory"
                    )

                    st.write(
                        f"🌱 **Crop:** {crop}"
                    )

                    st.write(
                        f"⚠️ **Risk:** {advice['risk']}"
                    )

                    st.subheader(
                        "🔧 Recommended Action"
                    )

                    st.success(
                        advice["action"]
                    )

                    st.subheader(
                        "🛡️ Prevention"
                    )

                    st.info(
                        advice["prevention"]
                    )

                    st.markdown("---")

                    st.subheader(
                        "👨‍🌾 Expert Recommendation"
                    )

                    st.write(
                        "If symptoms continue to spread or the "
                        "crop is economically important, consult "
                        "a local agricultural officer or qualified "
                        "crop expert before applying pesticides "
                        "or other treatments."
                    )

                    st.info(
                        "ℹ️ AgriDoctor AI provides AI-based "
                        "screening and general agricultural guidance. "
                        "It does not replace professional diagnosis."
                    )

                except Exception as e:

                    st.error(
                        "❌ Something went wrong while "
                        "analyzing the image."
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# SMART IRRIGATION
# =========================================================

elif page == "💧 Smart Irrigation":

    st.header(
        "🌦️ Weather & Smart Irrigation"
    )

    st.write(
        "Combine crop type, soil moisture and weather "
        "conditions to generate an irrigation recommendation."
    )

    st.markdown("---")

    # LOCATION

    location = st.text_input(
        "📍 Farm Location",
        placeholder="Example: Madanapalle"
    )

    # CROP

    crop = st.selectbox(
        "🌱 Select Crop",
        [
            "Tomato",
            "Potato",
            "Grape",
            "Chilli",
            "Groundnut",
            "Rice",
            "Other"
        ]
    )

    # SOIL

    soil_moisture = st.slider(
        "💧 Soil Moisture",
        min_value=0,
        max_value=100,
        value=50,
        help="0 = completely dry, 100 = completely wet"
    )

    if soil_moisture <= 30:

        soil_condition = "Dry"

    elif soil_moisture <= 60:

        soil_condition = "Normal"

    else:

        soil_condition = "Wet"

    # LANGUAGE

    language = st.selectbox(
        "👨‍🌾 Farmer Advisory Language",
        [
            "English",
            "తెలుగు"
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌱 Selected Crop"
        )

        st.markdown(
            f"# {crop}"
        )

    with col2:

        st.subheader(
            "💧 Soil Condition"
        )

        st.markdown(
            f"# {soil_condition}"
        )

    if st.button(
        "🌦️ Check Weather & Irrigation",
        type="primary"
    ):

        if not location.strip():

            st.warning(
                "📍 Please enter your farm location first."
            )

        else:

            with st.spinner(
                "🌦️ Fetching weather information..."
            ):

                try:

                    weather = get_weather(
                        location
                    )

                    if weather is None:

                        st.error(
                            "❌ Location not found. "
                            "Try a nearby city name."
                        )

                    else:

                        # ---------------------------------
                        # WEATHER
                        # ---------------------------------

                        st.success(
                            f"📍 Weather found for "
                            f"{weather['place']}"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "🌡️ Temperature",
                                f"{weather['temperature']} °C"
                            )

                        with col2:

                            st.metric(
                                "💧 Humidity",
                                f"{weather['humidity']}%"
                            )

                        with col3:

                            st.metric(
                                "🌧️ Rain Probability",
                                f"{weather['rain_probability']}%"
                            )

                        st.subheader(
                            "🌦️ Today's Weather"
                        )

                        st.write(
                            f"🌡️ Temperature range: "
                            f"{weather['min_temp']}–"
                            f"{weather['max_temp']} °C"
                        )

                        st.write(
                            f"🌧️ Forecast rainfall: "
                            f"{weather['rainfall']} mm"
                        )

                        st.write(
                            f"🌦️ Current rain: "
                            f"{weather['current_rain']} mm"
                        )

                        # ---------------------------------
                        # IRRIGATION DECISION
                        # ---------------------------------

                        decision, reason = (
                            irrigation_decision(
                                soil_moisture,
                                weather["rain_probability"],
                                weather["rainfall"]
                            )
                        )

                        st.markdown("---")

                        st.header(
                            "💧 Smart Irrigation Recommendation"
                        )

                        if decision == "DO NOT IRRIGATE":

                            st.error(
                                f"🚫 {decision}"
                            )

                        elif decision == "DELAY IRRIGATION":

                            st.warning(
                                f"⏳ {decision}"
                            )

                        elif decision == "IRRIGATE":

                            st.success(
                                f"💧 {decision}"
                            )

                        else:

                            st.info(
                                f"🌱 {decision}"
                            )

                        st.write(
                            reason
                        )

                        # ---------------------------------
                        # SUMMARY
                        # ---------------------------------

                        st.subheader(
                            "📋 Irrigation Decision Summary"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "🌱 Crop",
                                crop
                            )

                        with col2:

                            st.metric(
                                "💧 Soil",
                                soil_condition
                            )

                        with col3:

                            st.metric(
                                "🌧️ Rain",
                                f"{weather['rain_probability']}%"
                            )

                        # ---------------------------------
                        # FARMER ADVISORY
                        # ---------------------------------

                        st.markdown("---")

                        st.header(
                            "👨‍🌾 Farmer-Friendly Advisory"
                        )

                        message = farmer_message(
                            language,
                            crop,
                            decision,
                            weather["rain_probability"]
                        )

                        st.info(
                            "🌱 " + message
                        )

                        # ---------------------------------
                        # WHAT SHOULD FARMER DO?
                        # ---------------------------------

                        st.subheader(
                            "📢 What should the farmer do?"
                        )

                        if decision in [
                            "DO NOT IRRIGATE",
                            "DELAY IRRIGATION"
                        ]:

                            if language == "తెలుగు":

                                action = (
                                    "👉 ప్రస్తుతం నీరు పెట్టకుండా "
                                    "నేల తేమ మరియు వాతావరణాన్ని గమనించండి."
                                )

                            else:

                                action = (
                                    "👉 Do not add irrigation now. "
                                    "Monitor soil moisture and weather."
                                )

                        elif decision == "IRRIGATE":

                            if language == "తెలుగు":

                                action = (
                                    "👉 అవసరమైన మేరకు నీరు పెట్టండి. "
                                    "అధికంగా నీరు పెట్టవద్దు."
                                )

                            else:

                                action = (
                                    "👉 Irrigate moderately and avoid "
                                    "overwatering."
                                )

                        else:

                            if language == "తెలుగు":

                                action = (
                                    "👉 నేల తేమను కొనసాగిస్తూ "
                                    "క్రమం తప్పకుండా గమనించండి."
                                )

                            else:

                                action = (
                                    "👉 Continue monitoring soil moisture "
                                    "and weather conditions."
                                )

                        st.success(
                            action
                        )

                        # ---------------------------------
                        # VOICE
                        # ---------------------------------

                        st.subheader(
                            "🔊 Voice Advisory"
                        )

                        st.write(
                            "The farmer can listen to the recommendation."
                        )

                        with st.spinner(
                            "🔊 Preparing voice advisory..."
                        ):

                            audio = create_voice(
                                message,
                                language
                            )

                        if audio:

                            st.audio(
                                audio,
                                format="audio/mp3"
                            )

                            st.success(
                                "🔊 Voice advisory ready. "
                                "Press Play to listen."
                            )

                        else:

                            st.warning(
                                "Voice generation is currently unavailable."
                            )

                        st.info(
                            "ℹ️ Weather-based irrigation is an "
                            "advisory estimate. Actual irrigation "
                            "should also consider crop type, soil type, "
                            "crop growth stage, root-zone moisture "
                            "and local farm conditions."
                        )

                except Exception as e:

                    st.error(
                        "❌ Unable to retrieve weather data."
                    )

                    st.write(
                        "Please check your internet connection "
                        "and try again."
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🌱 AgriDoctor AI | SGDC Hackthon 2K26 | "
    "AET-03 Smart Agriculture: AI Crop Doctor + Autonomous Advisory"
)