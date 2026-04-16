import gradio as gr
import joblib
from keras.models import load_model

# -------------------------------
# ✅ Load model & vectorizer
# -------------------------------
try:
    model = load_model("model.keras")   # or change to model.h5 if needed
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Error loading files: {e}")

# -------------------------------
# ✅ Prediction function
# -------------------------------
def predict(message):
    if message.strip() == "":
        return "⚠️ Please enter a message"

    vec = vectorizer.transform([message]).toarray()
    pred = model.predict(vec)[0][0]

    confidence = pred * 100

    if pred > 0.5:
        return f"🚫 Spam Message\nConfidence: {confidence:.2f}%"
    else:
        return f"✅ Not Spam\nConfidence: {(100-confidence):.2f}%"

# -------------------------------
# ✅ UI
# -------------------------------
app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter your message here..."),
    outputs="text",
    title="📩 Smart Spam Detection",
    description="Check whether a message is Spam or Not"
)

# -------------------------------
# ✅ Run
# -------------------------------
if __name__ == "__main__":
    app.launch()