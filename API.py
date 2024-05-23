from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import urllib.request
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

model_image = load_model('diagram_classifier_and_scoring_model_with_VGG16.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy')

model_text = SentenceTransformer('bert-base-nli-mean-tokens')
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def calculate_bert_similarity(model_answer, student_answer):
    embedding1 = model_text.encode(model_answer, convert_to_tensor=True)
    embedding2 = model_text.encode(student_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

def calculate_cosine_similarity_preprocessed(model_answer, student_answer):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([model_answer, student_answer])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0, 1]

def preprocess_text(text):
    text = remove_tags(text)
    text = special_char(text)
    text = convert_lower(text)
    text = remove_punc(text)
    text = remove_stopwords(text)
    text = lemmatize_word(text)
    return text

def remove_tags(text):
    remove = re.compile(r'<.*?>')
    return re.sub(remove, '', text)

def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]

def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])

def convert_lower(text):
    return text.lower()

def remove_punc(txt):
    txt_nopunt="".join([c for c in txt if c not in string.punctuation])
    return txt_nopunt

def classify_similarity(similarity_score):
    for i, threshold in enumerate(thresholds, start=1):
        if similarity_score <= threshold:
            return i
    return len(thresholds) + 1

def preprocess_image(img_path):
    # Load the image
    img = cv2.imread(img_path)
    if img is not None:
        # Resize the image to 224x224
        img = cv2.resize(img, (224, 224))
        # Normalize the image
        img = img / 255.0
        # Convert to array and add batch dimension
        img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def process():
    req_data = request.get_json()
    data = req_data.get('data', [])
    
    results = []
    
    for item in data:
        if "model_answer" in item and "student_answer" in item:
            answerid=item['id']
            model_answer = item['model_answer']
            student_answer = item['student_answer']
            
            model_answer = preprocess_text(model_answer)
            student_answer = preprocess_text(student_answer)
            
            bert_score = calculate_bert_similarity(model_answer, student_answer)
            cosine_score = calculate_cosine_similarity_preprocessed(model_answer, student_answer)
            
            average_similarity = (bert_score + cosine_score) / 2
            
            classification = classify_similarity(average_similarity)
            
            results.append({"id":answerid,"type": "text_similarity", "similarity_classification": classification})
        
        elif "url" in item:
            imagid=item['id']
            img_url = item['url']
            
            try:
                # Download the image
                img_path = 'temp_image.png'
                urllib.request.urlretrieve(img_url, img_path)
                
                # Preprocess the image
                input_image = preprocess_image(img_path)
                
                # Make predictions
                predictions = model_image.predict(input_image)
                
                # Extract the classification and regression outputs
                classification_prediction = predictions[0]
                regression_prediction = predictions[1]
                
                # Decode the classification prediction
                predicted_class = np.argmax(classification_prediction, axis=1)
                predicted_class_label = label_encoder.inverse_transform(predicted_class)
                
                # Clean up the temporary image
                os.remove(img_path)
                
                # Append the results
                results.append({
                    "id":imagid,
                    "type": "image_classification",
                    "predicted_class": predicted_class_label[0],
                    "predicted_score": float(regression_prediction[0][0])
                })
            except Exception as e:
                results.append({"error": str(e)})
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
