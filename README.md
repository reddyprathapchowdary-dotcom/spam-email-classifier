# spam-email-classifier
spam message 
# 📩 Smart Spam Detection System (ANN + NLP)

##  Project Overview
This project is a **Spam Email/SMS Classifier** built using **Artificial Neural Networks (ANN)** and **Natural Language Processing (NLP)**.

The model predicts whether a given message is **Spam 🚫** or **Not Spam ✅**.

##  Technologies Used
 Python  
 TensorFlow / Keras  
 Scikit-learn  
 NLP (TF-IDF Vectorization)  
 Gradio (for UI)  

## 📊 Features
 Text preprocessing using TF-IDF  
 ANN-based classification model  
 High accuracy (~98%)  
 User-friendly web interface  
 Real-time prediction  

## 📁 Project Structure
smart-email-classifier/
│
├── train.py
├── app.py
├── spam.csv
├── model.keras
├── vectorizer.pkl
├── requirements.txt

## ⚙️ How to Run

### Step 1: Install dependencies
pip install -r requirements.txt
python train.py
## Step 2: Train the model
python train.py
## Step 3: Run the app
python app.py
## Output
Enter a message
Model predicts:
🚫 Spam
✅ Not Spam
## Example
Input:
Congratulations! You won ₹10,000. Click now!
## Output:
🚫 Spam Message
## Model Performance
Accuracy: ~98%
Dataset: SMS Spam Collection

https://huggingface.co/spaces/ReddyPrathap96/spam_email_classifier
