import os
import json
import math
import urllib.request
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
import torch
from werkzeug.utils import secure_filename
from markupsafe import escape
import glob, joblib
from bs4 import BeautifulSoup
import requests
import pickle
import spacy
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Model Comparison Charts
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from nltk.util import ngrams
import contractions
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel 
# Augmentation
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
# NLP and embeddings
import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nlp_spacy = spacy.load("en_core_web_sm")
# except OSError:
#     print("Downloading en_core_web_sm model for spaCy...")
#     spacy.cli.download("en_core_web_sm")
#     nlp_spacy = spacy.load("en_core_web_sm")
# MarianMTModel, MarianTokenizer (optional for back_translation)
nlp_spacy = spacy.load("en_core_web_sm")
# Initial setup
app = Flask(__name__)
app.secret_key = os.urandom(24) # More secure secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask-session' # Standard directory name
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models' # For saved models
app.config['STATIC_FOLDER'] = './static' # For charts

# Ensure directories exist
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'charts'), exist_ok=True) # Subfolder for charts

Session(app)

# Placeholder for models trained IN THE CURRENT SESSION (not necessarily saved to disk yet)
# This helps distinguish from models loaded from disk.
# session_trained_models = {} # We will primarily use flask session for simplicity here.

# --- Chatbot Knowledge Base ---
chatbot_kb = {
    "nlp là gì": "NLP (Natural Language Processing) là một lĩnh vực của trí tuệ nhân tạo giúp máy tính hiểu, diễn giải và tạo ra ngôn ngữ của con người.",
    "tăng cường dữ liệu": "Tăng cường dữ liệu (Data Augmentation) là kỹ thuật tạo ra các mẫu dữ liệu mới từ dữ liệu hiện có để tăng kích thước và sự đa dạng của tập huấn luyện.",
    "vector hóa": "Vector hóa (Vectorization) là quá trình chuyển đổi văn bản thành các vector số mà máy tính có thể xử lý. Ví dụ: TF-IDF, Word2Vec.",
    "ứng dụng này làm gì": "Ứng dụng này cho phép bạn thực hiện các tác vụ NLP cơ bản như nhập liệu, tăng cường, tiền xử lý, vector hóa, huấn luyện mô hình phân loại văn bản, gợi ý sản phẩm và chatbot.",
    "nhập liệu": "Tab 'Nhập liệu' dùng để đưa văn bản vào hệ thống, có thể nhập tay, tải file TXT, CSV hoặc cào từ web.",
    "tăng cường": "Tab 'Tăng cường' giúp bạn tạo thêm dữ liệu huấn luyện bằng các kỹ thuật như dịch ngược, thay thế đồng nghĩa,...",
    "xử lý": "Tab 'Xử lý' (Tiền xử lý) dùng để làm sạch và chuẩn hóa văn bản, ví dụ: tách từ, xóa stop words, viết thường,...",
    "biểu diễn": "Tab 'Biểu diễn' (Vector hóa) chuyển văn bản đã xử lý thành dạng số để máy học có thể hiểu được.",
    "train": "Tab 'Train' dùng để huấn luyện các mô hình phân loại văn bản (như Naive Bayes, SVM) trên dữ liệu CSV đã được vector hóa.",
    "gợi ý": "Tab 'Gợi ý sản phẩm' sử dụng mô hình SVD để gợi ý các sản phẩm tương tự dựa trên mô tả bạn nhập vào.",
    "chatbot": "Tab 'Chatbot' là nơi bạn đang tương tác đây! Tôi có thể trả lời các câu hỏi về NLP và ứng dụng này.",
    "naive bayes": "Naive Bayes là một thuật toán phân loại dựa trên định lý Bayes với giả định ngây thơ (naive) về tính độc lập giữa các đặc trưng.",
    "svm": "SVM (Support Vector Machine) là một thuật toán học có giám sát mạnh mẽ, tìm một siêu phẳng để phân tách tốt nhất các lớp dữ liệu.",
    "logistic regression": "Logistic Regression là một mô hình thống kê dùng để dự đoán xác suất của một biến nhị phân (ví dụ: có/không, đúng/sai).",
    "tf-idf": "TF-IDF (Term Frequency-Inverse Document Frequency) là một trọng số thống kê phản ánh tầm quan trọng của một từ đối với một tài liệu trong một bộ sưu tập hoặc ngữ liệu.",
    "word2vec": "Word2Vec là một kỹ thuật tạo ra các vector nhúng từ (word embeddings) bằng cách học các mối quan hệ giữa các từ trong một kho văn bản lớn."
}

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Đọc từ biến môi trường
gemini_model_instance = None # Đổi tên biến để tránh nhầm lẫn với tên module

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model_instance = genai.GenerativeModel('gemini-2.0-flash') # Sử dụng model flash
        print("Gemini Model configured successfully with gemini-2.0-flash.")
    except Exception as e:
        print(f"ERROR: Could not configure Gemini API: {e}")
        print("Please ensure your GOOGLE_API_KEY is correct and has access to the Gemini API.")
else:
    print("WARNING: GOOGLE_API_KEY not set in environment. Gemini chatbot will not function.")

# --- Back Translation (Optional, can be heavy) ---
# Helsinki-NLP models for translation
# try:
#     en_to_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
#     en_to_fr_tokenizer = MarianTokenizer.from_pretrained(en_to_fr_model_name)
#     en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr_model_name)
#     fr_to_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
#     fr_to_en_tokenizer = MarianTokenizer.from_pretrained(fr_to_en_model_name)
#     fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en_model_name)
#     BACK_TRANSLATION_ENABLED = True
# except Exception as e:
#     print(f"Could not load MarianMT models for back-translation: {e}. Back-translation will be disabled.")
#     BACK_TRANSLATION_ENABLED = False

# def translate_text_marian(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     translated_tokens = model.generate(**inputs, max_length=512)
#     return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# def back_translate_text(text, src_lang_model, src_lang_tokenizer, tgt_lang_model, tgt_lang_tokenizer):
#     if not BACK_TRANSLATION_ENABLED:
#         return f"[Back-translation disabled] {text}"
#     intermediate_translation = translate_text_marian(text, src_lang_model, src_lang_tokenizer)
#     final_translation = translate_text_marian(intermediate_translation, tgt_lang_model, tgt_lang_tokenizer)
#     return final_translation
# --- End Back Translation ---


# Entity replacement using spaCy
def entity_replacement(text, nlp_processor):
    doc = nlp_processor(text)
    new_text = text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            new_text = new_text.replace(ent.text, "[PERSON_REPLACED]") # Generic placeholder
        elif ent.label_ in ["ORG", "GPE", "LOC"]:
            new_text = new_text.replace(ent.text, f"[{ent.label_}_REPLACED]")
    return new_text

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def home():
    session.clear() # Clear session on new visit for a fresh start
    return render_template("index.html")

# --- INPUT TAB ---
@app.route('/upload_txt', methods=['POST'])
def upload_txt():
    if 'txt_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['txt_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, ['txt']):
        try:
            content = file.read().decode('utf-8')
            return jsonify({'content': content}), 200
        except Exception as e:
            return jsonify({'error': f'Error reading TXT file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type, only .txt allowed'}), 400

@app.route('/scrape_web', methods=['POST'])
def scrape_web():
    url = request.json.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Extract text from common tags, prioritizing main content if possible
        texts = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'div']):
            # Avoid script, style, nav, footer if possible by checking parent or class
            if element.name in ['script', 'style', 'nav', 'footer']:
                continue
            if any(parent.name in ['script', 'style', 'nav', 'footer'] for parent in element.parents):
                continue
            
            text_content = element.get_text(separator=' ', strip=True)
            if text_content and len(text_content) > 20: # Filter out very short/irrelevant text snippets
                 texts.append(text_content)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = [x for x in texts if not (x in seen or seen.add(x))]
        
        return jsonify({"scraped_text": "\n\n".join(unique_texts[:20])}) # Limit amount of text
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not fetch URL: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error scraping web page: {str(e)}"}), 500

@app.route('/upload_csv_input', methods=['POST']) # Renamed to avoid conflict
def upload_csv_input():
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, ['csv']):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            df = pd.read_csv(filepath)
            session['input_csv_path'] = filepath # Store path for later use
            session['input_csv_columns'] = df.columns.tolist()
            # Store a sample for display
            sample_data = df.head(3).to_dict(orient='records')
            return jsonify({'columns': df.columns.tolist(), 'sample': sample_data}), 200
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type, only .csv allowed'}), 400

@app.route('/select_csv_cols', methods=['POST'])
def select_csv_cols():
    data = request.json
    text_col = data.get('text_col')
    label_col = data.get('label_col')
    if not text_col or not label_col:
        return jsonify({'error': 'Text column and Label column must be selected'}), 400
    
    session['csv_text_col'] = text_col
    session['csv_label_col'] = label_col
    return jsonify({'message': f'Columns selected: Text={text_col}, Label={label_col}'}), 200

# --- AUGMENTATION TAB ---
@app.route('/augment_text', methods=['POST'])
def augment_text():
    data = request.json
    text = data.get("text", "")
    options = data.get("options", [])
    is_csv = data.get("is_csv", False)

    if not text and not is_csv: # For manual, text must be present
        return jsonify({"error": "No input text provided for manual augmentation."}), 400
    if is_csv and ('csv_text_col' not in session or 'input_csv_path' not in session):
        return jsonify({"error": "CSV data or text column not set up for CSV augmentation."}), 400

    results = []

    if is_csv:
        try:
            df = pd.read_csv(session['input_csv_path'])
            text_col = session['csv_text_col']
            # Process a sample of the CSV (e.g., first 5 rows) for demonstration
            texts_to_augment = df[text_col].astype(str).head(5).tolist() 
        except Exception as e:
            return jsonify({"error": f"Error reading CSV for augmentation: {str(e)}"}), 500
    else:
        texts_to_augment = [text]

    for original_text_item in texts_to_augment:
        augmented_item = original_text_item
        for opt in options:
            try:
                if opt == "back_translation":
                    # if BACK_TRANSLATION_ENABLED:
                    #    augmented_item = back_translate_text(augmented_item, en_to_fr_model, en_to_fr_tokenizer, fr_to_en_model, fr_to_en_tokenizer)
                    # else:
                    augmented_item = f"[BT_SKIPPED] {augmented_item}" # Placeholder if heavy models not loaded
                elif opt == "synonym_replacement":
                    aug = naw.SynonymAug(aug_src="wordnet")
                    augmented_item = aug.augment(augmented_item)[0] if aug.augment(augmented_item) else augmented_item
                elif opt == "random_insertion":
                    aug = naw.RandomWordAug(action="insert") # Changed from substitute to insert
                    augmented_item = aug.augment(augmented_item)[0] if aug.augment(augmented_item) else augmented_item
                elif opt == "random_swap":
                    aug = naw.RandomWordAug(action="swap")
                    augmented_item = aug.augment(augmented_item)[0] if aug.augment(augmented_item) else augmented_item
                elif opt == "random_deletion":
                    aug = naw.RandomWordAug(action="delete")
                    augmented_item = aug.augment(augmented_item)[0] if aug.augment(augmented_item) else augmented_item
                elif opt == "entity_replacement":
                    augmented_item = entity_replacement(augmented_item, nlp_spacy)
                elif opt == "add_noise": # Character noise
                    aug = nac.RandomCharAug(action="insert")
                    augmented_item = aug.augment(augmented_item)[0] if aug.augment(augmented_item) else augmented_item
            except Exception as e:
                print(f"Error during augmentation option '{opt}': {e}") # Log error
                # Keep processing with other options or original text
        
        if is_csv:
            results.append({"original": original_text_item, "augmented": augmented_item})
        else: # Manual
            results.append({"augmented": augmented_item}) # Only need augmented for manual

    return jsonify({"results": results, "is_csv": is_csv})


# --- PREPROCESSING TAB ---
@app.route('/preprocess_text', methods=['POST'])
def preprocess_text():
    data = request.json
    text = data.get("text", "")
    options = data.get("options", [])
    is_csv = data.get("is_csv", False)

    if not text and not is_csv:
        return jsonify({"error": "No input text for manual preprocessing."}), 400
    if is_csv and ('csv_text_col' not in session or 'input_csv_path' not in session):
        return jsonify({"error": "CSV data or text column not set up for CSV preprocessing."}), 400

    processed_results = []
    
    if is_csv:
        try:
            df = pd.read_csv(session['input_csv_path'])
            text_col = session['csv_text_col']
            # Process a sample (e.g., first 5 rows)
            texts_to_process = df[text_col].astype(str).head(5).tolist()
        except Exception as e:
            return jsonify({"error": f"Error reading CSV for preprocessing: {str(e)}"}), 500
    else:
        texts_to_process = [text]

    for original_text_item in texts_to_process:
        result_item = original_text_item
        is_tokenized_list = False # Flag to track if result_item is a list of tokens

        for opt in options:
            try:
                if opt == "sentence_tokenization":
                    if not is_tokenized_list: # Only sentence tokenize if not already word tokenized
                        result_item = [sent.text for sent in nlp_spacy(str(result_item)).sents]
                    # If already word_tokenized, sentence tokenization is less meaningful in this flow
                elif opt == "word_tokenization":
                    if isinstance(result_item, list) and all(isinstance(s, str) for s in result_item): # Was sentence_tokenized
                        result_item = [token.text for sent in result_item for token in nlp_spacy(sent)]
                    else: # Is a single string
                        result_item = [token.text for token in nlp_spacy(str(result_item))]
                    is_tokenized_list = True
                elif opt == "remove_stopwords":
                    if not is_tokenized_list: result_item = [token.text for token in nlp_spacy(str(result_item))] # Tokenize if not already
                    stopwords = nltk.corpus.stopwords.words('english') # Assuming English for now
                    result_item = [word for word in result_item if word.lower() not in stopwords]
                    is_tokenized_list = True
                elif opt == "rm_pun":
                    if not is_tokenized_list: result_item = [token.text for token in nlp_spacy(str(result_item))]
                    result_item = [word for word in result_item if not nlp_spacy(word)[0].is_punct]
                    is_tokenized_list = True
                elif opt == "lowercasing":
                    if not is_tokenized_list: result_item = [token.text for token in nlp_spacy(str(result_item))]
                    result_item = [word.lower() for word in result_item]
                    is_tokenized_list = True
                elif opt == "fix_abbreviations": # Best applied on strings or before aggressive tokenization
                    if is_tokenized_list:
                        result_item = [contractions.fix(word) for word in result_item]
                    else:
                        result_item = contractions.fix(str(result_item))
            except Exception as e:
                 print(f"Error during preprocessing option '{opt}': {e}")

        # If operations resulted in a list of tokens/sentences, join them for consistent output
        if isinstance(result_item, list):
            result_item = " ".join(result_item)
            
        if is_csv:
            processed_results.append({"original": original_text_item, "processed": result_item})
        else:
            processed_results.append({"processed": result_item})

    return jsonify({"results": processed_results, "is_csv": is_csv})

# --- VECTORIZATION TAB ---
@app.route('/vectorize_manual_text', methods=['POST'])
def vectorize_manual_text():
    data = request.json
    text_input = data.get("text", "")
    # selected_methods_list giờ là mảng một phần tử, ví dụ: ["tfidf"]
    selected_methods_list = data.get("methods", []) 
    if not text_input:
        return jsonify({"error": "No text provided for manual vectorization."}), 400
    if not selected_methods_list:
        return jsonify({"error": "No method selected for manual vectorization."}), 400

    # Lấy method duy nhất từ danh sách
    method_to_apply = selected_methods_list[0]
    
    tokens = [token.text for token in nlp_spacy(text_input.lower()) if not token.is_punct and not token.is_space]
    corpus = [" ".join(tokens)] 
    output_vectors = {} # Vẫn dùng output_vectors để chứa kết quả cho method đó

    BERT_ENABLED = False # Giữ BERT tắt cho nhẹ

    # Chỉ áp dụng method_to_apply
    try:
        if method_to_apply == "one_hot":
            # ... (logic cho one_hot như cũ)
            vocab = {word: i for i, word in enumerate(sorted(list(set(tokens))))}
            if not vocab: 
                output_vectors[method_to_apply] = {"error": "No tokens to one-hot encode."}
            else:
                one_hot_result = {}
                for token in tokens:
                    vec = np.zeros(len(vocab))
                    if token in vocab: vec[vocab[token]] = 1
                    one_hot_result[token] = vec.tolist()
                output_vectors[method_to_apply] = one_hot_result
        elif method_to_apply == "bow":
            if not corpus[0]: 
                output_vectors[method_to_apply] = {"error": "No text for BoW."}
            else:
                vectorizer = CountVectorizer()
                bow_matrix = vectorizer.fit_transform(corpus).toarray()
                output_vectors[method_to_apply] = {"vector": bow_matrix[0].tolist(), "features": vectorizer.get_feature_names_out().tolist()}
        elif method_to_apply == "tfidf":
            if not corpus[0]:
                output_vectors[method_to_apply] = {"error": "No text for TF-IDF."}
            else:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
                output_vectors[method_to_apply] = {"vector": tfidf_matrix[0].tolist(), "features": vectorizer.get_feature_names_out().tolist()}
        elif method_to_apply == "ngram":
            bigrams = list(ngrams(tokens, 2))
            output_vectors[method_to_apply] = [" ".join(gram) for gram in bigrams]
        elif method_to_apply == "word2vec":
            # ... (logic cho word2vec như cũ)
            if not tokens: 
                output_vectors[method_to_apply] = {"error": "No tokens for Word2Vec."}
            else:
                model = Word2Vec([tokens], vector_size=10, window=2, min_count=1, workers=1)
                w2v_vectors = {word: model.wv[word].tolist() for word in model.wv.index_to_key}
                output_vectors[method_to_apply] = w2v_vectors
        elif method_to_apply == "fasttext":
            # ... (logic cho fasttext như cũ)
            if not tokens: 
                output_vectors[method_to_apply] = {"error": "No tokens for FastText."}
            else:
                model = FastText([tokens], vector_size=10, window=2, min_count=1, workers=1)
                ft_vectors = {word: model.wv[word].tolist() for word in model.wv.index_to_key}
                output_vectors[method_to_apply] = ft_vectors
        elif method_to_apply == "bert" and BERT_ENABLED:
            pass # Giữ nguyên nếu BERT_ENABLED là False
        elif method_to_apply == "bert" and not BERT_ENABLED:
             output_vectors[method_to_apply] = {"error": "BERT model not loaded/enabled for manual demo."}
        else:
            output_vectors[method_to_apply] = {"error": f"Method '{method_to_apply}' not recognized for manual vectorization."}

    except Exception as e:
        output_vectors[method_to_apply] = {"error": f"Error vectorizing with {method_to_apply}: {str(e)}"}
            
    return jsonify({"vectors": output_vectors}) # Vẫn trả về cấu trúc "vectors" chứa 1 key


@app.route('/vectorize_csv_data', methods=['POST'])
def vectorize_csv_data():
    data = request.json
    # selected_methods giờ là mảng một phần tử do JS gửi lên
    selected_methods_list = data.get("methods", []) 
    
    if not selected_methods_list:
        return jsonify({"error": "No vectorization method selected."}), 400
    
    primary_method = selected_methods_list[0] # Lấy phương pháp duy nhất

    if 'input_csv_path' not in session or \
       'csv_text_col' not in session or \
       'csv_label_col' not in session:
        return jsonify({"error": "CSV data, text column, or label column not set up."}), 400

    try:
        df = pd.read_csv(session['input_csv_path'])
        texts = df[session['csv_text_col']].astype(str).fillna("").tolist()
        labels = df[session['csv_label_col']].tolist()
    except Exception as e:
        return jsonify({"error": f"Error reading CSV for vectorization: {str(e)}"}), 500

    if not texts:
        return jsonify({"error": "No text data found in selected CSV column."}), 400

    vectorizer_obj = None
    X_transformed = None

    # Các phương pháp được hỗ trợ cho training CSV
    supported_training_methods = ["tfidf", "bow"]

    if primary_method == "tfidf":
        vectorizer_obj = TfidfVectorizer(max_features=5000)
        X_transformed = vectorizer_obj.fit_transform(texts)
    elif primary_method == "bow":
        vectorizer_obj = CountVectorizer(max_features=5000)
        X_transformed = vectorizer_obj.fit_transform(texts)
    # elif primary_method == "one_hot": # Ví dụ nếu muốn hỗ trợ one-hot (cần code cẩn thận)
        # return jsonify({"error": "One-hot encoding for full CSV training is complex and not fully implemented here. Please choose TF-IDF or BoW."}), 400
    else:
        # Thông báo lỗi rõ ràng cho các phương pháp không được hỗ trợ cho training
        return jsonify({"error": f"Method '{primary_method}' is not currently supported as a primary vectorizer for CSV training in this demo. Please choose TF-IDF or BoW."}), 400

    session['vectorized_X_for_training'] = X_transformed.toarray().tolist()
    session['vectorized_y_for_training'] = labels
    session['active_vectorizer_name'] = primary_method
    session['active_vectorizer_object_bytes'] = pickle.dumps(vectorizer_obj) # Đã sửa thành pickle

    # --- Generate samples for display (chỉ cho primary_method) ---
    display_samples_dict = {}
    sample_texts_for_display = texts[:3]
    current_sample_vectors_for_display = []

    try:
        # Dùng lại vectorizer_obj đã fit để transform sample
        # (Lưu ý: vectorizer_obj.transform trả về sparse matrix, cần toarray())
        if hasattr(vectorizer_obj, 'transform'):
            sample_matrix_transformed = vectorizer_obj.transform(sample_texts_for_display).toarray()
            for row in sample_matrix_transformed:
                current_sample_vectors_for_display.append(row[:10].tolist()) # Lấy 10 chiều đầu
        else: # Fallback nếu vectorizer không có transform (ít xảy ra với TFIDF/BOW)
            for _ in sample_texts_for_display: current_sample_vectors_for_display.append(["N/A (transform error)"])
        
        display_samples_dict[primary_method] = current_sample_vectors_for_display
    except Exception as e_sample:
        display_samples_dict[primary_method] = [[f"Error generating sample: {str(e_sample)}"]]


    return jsonify({
        "message": f"CSV vectorized using '{primary_method}' for training. Samples displayed.",
        "training_vectorizer": primary_method,
        "num_samples_vectorized": len(texts),
        "dimension_for_training": X_transformed.shape[1],
        "display_samples": display_samples_dict, 
        "sample_original_texts": sample_texts_for_display
    })

# --- TRAINING TAB ---
@app.route('/train_model_from_csv', methods=['POST'])
def train_model_from_csv():
    data = request.json
    model_type = data.get("model_type", "nb")
    test_size = float(data.get("test_size", 0.2))
    model_params = data.get("params", {})

    if 'vectorized_X_for_training' not in session or 'vectorized_y_for_training' not in session:
        return jsonify({"error": "Vectorized data not found in session. Please vectorize CSV first."}), 400

    X = np.array(session['vectorized_X_for_training'])
    y = np.array(session['vectorized_y_for_training'])
    
    if 'active_vectorizer_object_bytes' not in session:
        return jsonify({"error": "Vectorizer object not found in session."}), 400
    vectorizer = pickle.loads(session['active_vectorizer_object_bytes'])
    vectorizer_name = session.get('active_vectorizer_name', 'unknown_vectorizer')

    if X.shape[0] != y.shape[0]:
        return jsonify({"error": "Mismatch in number of samples for X and y."}), 400
    if X.shape[0] == 0:
        return jsonify({"error": "No data to train on."}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    try:
        valid_params = {}
        if model_type == "logistic":
            allowed_log_params = {'C', 'penalty', 'solver', 'max_iter'}
            valid_params = {k: (float(v) if k == 'C' else (int(v) if k == 'max_iter' else v)) for k, v in model_params.items() if k in allowed_log_params and v}
            if 'max_iter' not in valid_params: valid_params['max_iter'] = 1000
            model = LogisticRegression(**valid_params)
        elif model_type == "svm":
            allowed_svm_params = {'C', 'kernel', 'gamma', 'degree'}
            valid_params = {}
            for k, v_str in model_params.items():
                if k in allowed_svm_params and v_str:
                    if k == 'C':
                        try: valid_params[k] = float(v_str)
                        except ValueError: print(f"Warning: SVM C value '{v_str}' invalid.")
                    elif k == 'gamma':
                        if v_str.lower() in ['scale', 'auto']: valid_params[k] = v_str.lower()
                        else:
                            try: valid_params[k] = float(v_str)
                            except ValueError: print(f"Warning: SVM gamma value '{v_str}' invalid.")
                    elif k == 'degree':
                        try: valid_params[k] = int(v_str)
                        except ValueError: print(f"Warning: SVM degree value '{v_str}' invalid.")
                    elif k == 'kernel': valid_params[k] = v_str
            model = SVC(probability=True, random_state=42, **valid_params)
        elif model_type == "knn":
            allowed_knn_params = {'n_neighbors', 'weights', 'algorithm'}
            valid_params = {k: (int(v) if k == 'n_neighbors' else v) for k, v in model_params.items() if k in allowed_knn_params and v}
            model = KNeighborsClassifier(**valid_params)
        elif model_type == "tree":
            allowed_tree_params = {'criterion', 'splitter', 'max_depth'}
            valid_params = {}
            for k, v_str in model_params.items():
                if k in allowed_tree_params and v_str:
                    if k == 'max_depth':
                        if v_str.strip().lower() == 'none' or v_str.strip() == '': # Kiểm tra "None" hoặc rỗng
                            valid_params[k] = None
                        else:
                            try: valid_params[k] = int(v_str)
                            except ValueError: print(f"Warning: Tree max_depth '{v_str}' invalid.")
                    elif k == 'criterion' or k == 'splitter':
                        valid_params[k] = v_str
            model = DecisionTreeClassifier(random_state=42, **valid_params)
        elif model_type == "rf": # Random Forest
            allowed_rf_params = {'n_estimators', 'criterion', 'max_depth'}
            valid_params = {}
            for k, v_str in model_params.items():
                if k in allowed_rf_params and v_str:
                    if k == 'n_estimators':
                        try: valid_params[k] = int(v_str)
                        except ValueError: print(f"Warning: RF n_estimators '{v_str}' invalid.")
                    elif k == 'max_depth':
                        if v_str.strip().lower() == 'none' or v_str.strip() == '': # Kiểm tra "None" hoặc rỗng
                            valid_params[k] = None
                        else:
                            try: valid_params[k] = int(v_str)
                            except ValueError: print(f"Warning: RF max_depth '{v_str}' invalid.")
                    elif k == 'criterion':
                         valid_params[k] = v_str
            model = RandomForestClassifier(random_state=42, **valid_params)
        elif model_type == "gb": # Gradient Boosting
            allowed_gb_params = {'n_estimators', 'learning_rate', 'max_depth'}
            valid_params = {}
            for k, v_str in model_params.items():
                if k in allowed_gb_params and v_str:
                    if k == 'n_estimators' or k == 'max_depth':
                        try: valid_params[k] = int(v_str)
                        except ValueError: print(f"Warning: GB {k} '{v_str}' invalid.")
                    elif k == 'learning_rate':
                        try: valid_params[k] = float(v_str)
                        except ValueError: print(f"Warning: GB learning_rate '{v_str}' invalid.")
            model = GradientBoostingClassifier(random_state=42, **valid_params)
        elif model_type == "nb":
            model = MultinomialNB()
        else:
            return jsonify({"error": "Invalid model type"}), 400

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        model_id = f"{model_type}_{vectorizer_name}_{pd.Timestamp.now().strftime('%H%M%S')}"
        session['last_trained_model_details'] = {
            'id': model_id,
            'model_type': model_type,
            'vectorizer_name': vectorizer_name,
            'model_bytes': pickle.dumps(model),
            'vectorizer_bytes': pickle.dumps(vectorizer),
            'accuracy': accuracy, 
            'report': report, 
            'X_test_bytes': pickle.dumps(X_test),
            'y_test_bytes': pickle.dumps(y_test),
            'y_pred_bytes': pickle.dumps(y_pred)
        }
        
        cm_filename = f"cm_{model_id}.png"
        cm_path_static = os.path.join('charts', cm_filename)
        cm_path_full = os.path.join(app.config['STATIC_FOLDER'], cm_path_static)
        
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_ if hasattr(model, 'classes_') else np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_ if hasattr(model, 'classes_') else np.unique(y_test))
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_id}")
        plt.tight_layout()
        fig.savefig(cm_path_full)
        plt.close(fig)

        return jsonify({
            "message": "Model trained successfully!",
            "model_id_for_saving": model_id,
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix_url": f"/static/{cm_path_static}?t={pd.Timestamp.now().timestamp()}"
        })

    except Exception as e:
        return jsonify({"error": f"Error training model: {str(e)}"}), 500


@app.route('/save_trained_model', methods=['POST'])
def save_trained_model_route():
    data = request.json
    user_model_name = data.get("user_model_name")
    
    if 'last_trained_model_details' not in session:
        return jsonify({"error": "No model details found in session to save."}), 400
    if not user_model_name:
        return jsonify({"error": "Please provide a name for the model."}), 400

    model_details = session['last_trained_model_details']
    disk_model_name = f"{user_model_name.replace(' ', '_')}_{model_details['id']}"
    
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"{disk_model_name}_model.joblib")
    vectorizer_path = os.path.join(app.config['MODEL_FOLDER'], f"{disk_model_name}_vectorizer.joblib")
    metadata_path = os.path.join(app.config['MODEL_FOLDER'], f"{disk_model_name}_metadata.json")

    try:
        # Lấy lại đối tượng model và vectorizer từ bytes trong session
        model_object = pickle.loads(model_details['model_bytes'])
        vectorizer_object = pickle.loads(model_details['vectorizer_bytes'])

        # Lưu đối tượng model và vectorizer bằng joblib.dump
        joblib.dump(model_object, model_path)
        joblib.dump(vectorizer_object, vectorizer_path)

        # ... (phần metadata giữ nguyên)
        metadata = {
            'user_model_name': user_model_name,
            'disk_model_name': disk_model_name,
            'model_type': model_details['model_type'],
            'vectorizer_name': model_details['vectorizer_name'],
            'accuracy': model_details['accuracy'], 
            'report': model_details['report'],
            'timestamp': pd.Timestamp.now().isoformat(),
            'cm_filename_part': f"cm_{model_details['id']}.png"
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return jsonify({"message": f"Model '{user_model_name}' saved successfully as {disk_model_name}."})
    except Exception as e:
        return jsonify({"error": f"Error saving model: {str(e)}"}), 500


@app.route('/get_saved_models_list', methods=['GET'])
def get_saved_models_list():
    saved_models = []
    for metadata_file in glob.glob(os.path.join(app.config['MODEL_FOLDER'], "*_metadata.json")):
        try:
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                saved_models.append({
                    "user_name": meta.get('user_model_name', 'Unknown'),
                    "disk_name": meta.get('disk_model_name', 'unknown_disk_name'), # This is the key for loading
                    "type": meta.get('model_type', 'N/A'),
                    "accuracy": meta.get('accuracy', 'N/A')
                })
        except Exception as e:
            print(f"Error reading metadata file {metadata_file}: {e}")
    return jsonify({"saved_models": sorted(saved_models, key=lambda x: x['user_name'])}), 200


@app.route('/predict_with_saved_model', methods=['POST'])
def predict_with_saved_model():
    data = request.json
    disk_model_name = data.get('disk_model_name')
    text_to_predict = data.get('text')

    if not disk_model_name or not text_to_predict:
        return jsonify({"error": "Model name or text not provided."}), 400

    model_path = os.path.join(app.config['MODEL_FOLDER'], f"{disk_model_name}_model.joblib")
    vectorizer_path = os.path.join(app.config['MODEL_FOLDER'], f"{disk_model_name}_vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return jsonify({"error": f"Model files for '{disk_model_name}' not found."}), 400

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess text slightly (e.g., lowercasing, basic cleaning) similar to training
        # This step should ideally match the preprocessing pipeline used before vectorization during training.
        # For simplicity, a basic cleaning:
        processed_text = " ".join([token.lemma_.lower() for token in nlp_spacy(text_to_predict) if not token.is_stop and not token.is_punct])

        text_vector = vectorizer.transform([processed_text]) # Must be a list/iterable
        prediction = model.predict(text_vector)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vector)
            # Map probabilities to classes
            prob_dict = {str(cls): prob for cls, prob in zip(model.classes_, probabilities[0])}


        return jsonify({
            "prediction": str(prediction[0]), # Ensure JSON serializable
            "probabilities": prob_dict if probabilities is not None else "N/A"
        })
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


@app.route('/generate_model_comparisons', methods=['POST'])
def generate_model_comparisons():
    saved_models_metadata_files = glob.glob(os.path.join(app.config['MODEL_FOLDER'], "*_metadata.json"))
    if not saved_models_metadata_files:
        return jsonify({"error": "No saved models found to compare."}), 400

    accuracies = []
    cm_urls = []

    for metadata_file in saved_models_metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
            accuracies.append((meta.get('user_model_name', meta.get('disk_model_name')), meta.get('accuracy', 0)))
            
            # Link to existing CM if available
            cm_chart_path_part = meta.get('cm_filename_part')
            if cm_chart_path_part and os.path.exists(os.path.join(app.config['STATIC_FOLDER'], 'charts', cm_chart_path_part)):
                 cm_urls.append({
                     "name": meta.get('user_model_name', meta.get('disk_model_name')),
                     "url": f"/static/charts/{cm_chart_path_part}?t={pd.Timestamp.now().timestamp()}"
                 })
            # Else: could re-generate CM if X_test, y_test, y_pred were saved, but simpler to rely on existing.

        except Exception as e:
            print(f"Error processing metadata {metadata_file} for comparison: {e}")

    if not accuracies:
        return jsonify({"error": "No valid model metadata found for comparison."}), 400

    accuracies.sort(key=lambda x: x[1], reverse=True) # Sort by accuracy
    names = [item[0] for item in accuracies]
    values = [item[1] for item in accuracies]

    # Accuracy Comparison Bar Chart
    acc_chart_filename = f"accuracy_comparison_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.png"
    acc_chart_path_static = os.path.join('charts', acc_chart_filename)
    acc_chart_path_full = os.path.join(app.config['STATIC_FOLDER'], acc_chart_path_static)
    
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 5)) # Dynamic width
    ax.bar(names, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0, 1.05) # Accuracy range
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(acc_chart_path_full)
    plt.close(fig)

    return jsonify({
        "accuracy_chart_url": f"/static/{acc_chart_path_static}?t={pd.Timestamp.now().timestamp()}",
        "confusion_matrices": cm_urls
    })

@app.route('/clear_all_saved_models_and_charts', methods=['POST'])
def clear_all_saved_models_and_charts():
    try:
        # Delete model files
        model_files = glob.glob(os.path.join(app.config['MODEL_FOLDER'], "*.joblib"))
        metadata_files = glob.glob(os.path.join(app.config['MODEL_FOLDER'], "*.json"))
        for f_path in model_files + metadata_files:
            os.remove(f_path)
        
        # Delete chart files
        chart_files = glob.glob(os.path.join(app.config['STATIC_FOLDER'], 'charts', "*.png"))
        for f_path in chart_files:
            os.remove(f_path)
        
        session.pop('last_trained_model_details', None) # Clear any lingering trained model in session

        return jsonify({"message": "All saved models and charts have been cleared."})
    except Exception as e:
        return jsonify({"error": f"Error clearing data: {str(e)}"}), 500


# --- RECOMMENDATION TAB (SVD) ---
@app.route('/upload_recommend_dataset', methods=['POST'])
def upload_recommend_dataset():
    if 'recommend_csv_file' not in request.files: # Ensure unique name for file input in JS
        return jsonify({'error': 'No file part'}), 400
    file = request.files['recommend_csv_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, ['csv']):
        filename = secure_filename(f"recommend_data_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            df = pd.read_csv(filepath)
            session['recommend_csv_path'] = filepath
            return jsonify({'columns': df.columns.tolist()}), 200
        except Exception as e:
            return jsonify({'error': f'Error reading recommendation CSV: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type for recommendation dataset, only .csv allowed'}), 400

@app.route('/confirm_recommend_cols', methods=['POST'])
def confirm_recommend_cols():
    data = request.json
    session['rec_keyword_col'] = data.get('keyword_col')
    session['rec_product_id_col'] = data.get('product_id_col')
    if not session['rec_keyword_col'] or not session['rec_product_id_col']:
        return jsonify({'error': 'Keyword and Product ID columns must be selected.'}), 400
    return jsonify({'message': 'Recommendation columns confirmed.'})

@app.route('/train_svd_recommender', methods=['POST'])
def train_svd_recommender():
    if 'recommend_csv_path' not in session or \
       'rec_keyword_col' not in session or \
       'rec_product_id_col' not in session:
        return jsonify({"error": "Recommendation dataset or columns not set up."}), 400

    try:
        df = pd.read_csv(session['recommend_csv_path'])
        keyword_col = session['rec_keyword_col']
        product_id_col = session['rec_product_id_col']

        # Ensure columns exist
        if keyword_col not in df.columns or product_id_col not in df.columns:
             return jsonify({"error": f"Selected columns ('{keyword_col}', '{product_id_col}') not found in CSV."}), 400


        product_descriptions = df[keyword_col].astype(str).fillna("").tolist()
        product_ids = df[product_id_col].tolist() # Or any other relevant product info

        if not product_descriptions:
             return jsonify({"error": "No product descriptions found in selected column."}), 400


        # 1. TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)

        # 2. TruncatedSVD
        # n_components should be less than n_features from TF-IDF
        n_components = min(100, tfidf_matrix.shape[1] - 1) 
        if n_components <=0:
             return jsonify({"error": f"Not enough features ({tfidf_matrix.shape[1]}) from TF-IDF to perform SVD. Try with more diverse data or adjust TF-IDF parameters."}), 400
        
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd_model.fit_transform(tfidf_matrix)

        # Store models and data in session
        session['rec_tfidf_vectorizer_bytes'] = pickle.dumps(tfidf_vectorizer)
        session['rec_svd_model_bytes'] = pickle.dumps(svd_model)
        session['rec_latent_matrix_bytes'] = pickle.dumps(latent_matrix)
        session['rec_product_ids'] = product_ids # Store original product IDs/names
        session['rec_product_descriptions'] = product_descriptions # Store original descriptions

        return jsonify({"message": f"SVD Recommender trained with {n_components} components. Ready for queries."})

    except Exception as e:
        return jsonify({"error": f"Error training SVD recommender: {str(e)}"}), 500


@app.route('/get_svd_recommendations', methods=['POST'])
def get_svd_recommendations():
    data = request.json
    query_text = data.get("query_text", "")

    if not query_text:
        return jsonify({"error": "Please enter a query text."}), 400

    required_sessions = ['rec_tfidf_vectorizer_bytes', 'rec_svd_model_bytes', 
                         'rec_latent_matrix_bytes', 'rec_product_ids', 'rec_product_descriptions']
    if not all(key in session for key in required_sessions):
        return jsonify({"error": "SVD Recommender models not found in session. Please train first."}), 400

    try:
        tfidf_vectorizer = pickle.loads(session['rec_tfidf_vectorizer_bytes'])
        svd_model = pickle.loads(session['rec_svd_model_bytes'])
        latent_matrix_all_products = pickle.loads(session['rec_latent_matrix_bytes'])
        product_ids = session['rec_product_ids']
        product_descriptions = session['rec_product_descriptions']

        # Transform query
        query_tfidf = tfidf_vectorizer.transform([query_text])
        query_latent = svd_model.transform(query_tfidf)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_latent, latent_matrix_all_products)
        
        # Get top N recommendations
        top_n_indices = similarities[0].argsort()[::-1][:5] # Top 5

        recommendations = []
        for index in top_n_indices:
            recommendations.append({
                "product_id": str(product_ids[index]), # Ensure serializable
                "description_preview": product_descriptions[index][:100] + "...", # Show a preview
                "similarity_score": float(similarities[0][index]) # Ensure serializable
            })
        
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": f"Error getting recommendations: {str(e)}"}), 500


# --- CHATBOT TAB ---
@app.route('/chat_message', methods=['POST'])
def chat_message():
    user_message = request.json.get("message", "").lower().strip()
    if not user_message:
        return jsonify({"reply": "Xin lỗi, tôi không nhận được tin nhắn của bạn."}), 400

    # 1. Check Knowledge Base (simple keyword matching)
    for keyword, answer in chatbot_kb.items():
        if keyword in user_message: # Simple substring check
            return jsonify({"reply": answer})
    
    # 2. (Optional) Fallback to a more advanced model like Gemini
    # For this example, let's keep it simple. If you have Gemini API key:
    # try:
    #     api_key = "YOUR_GEMINI_API_KEY" # Replace with your actual key
    #     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    #     headers = {"Content-Type": "application/json"}
    #     # You can prepend a system prompt to guide Gemini
    #     # system_prompt = "You are a helpful assistant for an NLP web application. Answer questions about NLP or the application's features. If you don't know, say so."
    #     # full_prompt = f"{system_prompt}\nUser: {user_message}\nAssistant:"
    #     payload = {
    #         "contents": [{"parts": [{"text": user_message}]}] # Using original user_message
    #     }
    #     req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
    #     with urllib.request.urlopen(req) as response:
    #         if response.status == 200:
    #             result = json.loads(response.read().decode())
    #             # Safely access the reply
    #             if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
    #                 reply = result["candidates"][0]["content"]["parts"][0]["text"]
    #                 return jsonify({"reply": reply})
    #             else:
    #                 return jsonify({"reply": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu qua Gemini."})
    #         else:
    #             return jsonify({"reply": f"Lỗi từ Gemini API: {response.status}"})
    # except Exception as e:
    #     print(f"Gemini API error: {e}")
    #     return jsonify({"reply": "Xin lỗi, tôi không thể kết nối tới dịch vụ chatbot nâng cao ngay lúc này."})
    
    # Fallback if no KB match and no Gemini
    return jsonify({"reply": "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể hỏi về các chức năng của ứng dụng hoặc các khái niệm NLP cơ bản không?"})


# --- Endpoint cho Chatbot Gemini ---
@app.route('/gemini_chat_message', methods=['POST'])
def gemini_chat_message_route():
    if not gemini_model_instance:
        return jsonify({"reply": "Gemini API chưa được cấu hình (thiếu API Key hoặc lỗi cấu hình)."}), 503

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Xin lỗi, tôi không nhận được tin nhắn của bạn."}), 400

    # --- ĐỊNH NGHĨA SYSTEM PROMPT Ở ĐÂY ---
    system_prompt = f"""
Bạn là một trợ lý AI hữu ích được tích hợp vào một Ứng dụng Web NLP Pro.
Ứng dụng này (NLP WebApp Pro) có các chức năng chính sau:

1.  **Nhập liệu (Input Tab):**
    *   Nhập văn bản thủ công.
    *   Tải file TXT.
    *   Tải file CSV và chọn cột văn bản, cột nhãn.
    *   Cào dữ liệu văn bản từ một URL web.

2.  **Tăng cường dữ liệu (Augment Tab):**
    *   Áp dụng cho văn bản thủ công hoặc dữ liệu từ cột văn bản của file CSV.
    *   Các kỹ thuật: Dịch ngược, Thay thế đồng nghĩa, Thêm từ ngẫu nhiên, Đổi chỗ từ, Xóa từ ngẫu nhiên, Thay thế thực thể (ví dụ: PERSON thành [PERSON_REPLACED]), Thêm nhiễu ký tự.

3.  **Tiền xử lý (Preprocess Tab):**
    *   Áp dụng cho văn bản thủ công hoặc dữ liệu từ cột văn bản của file CSV.
    *   Các bước: Tách câu (Sentence tokenization), Tách từ (Word tokenization), Xóa stop words (tiếng Anh), Xóa dấu câu, Viết thường, Sửa lỗi viết tắt (contractions).

4.  **Biểu diễn dữ liệu (Vectorize Tab):**
    *   Chuyển đổi văn bản thành vector số.
    *   Cho văn bản thủ công: Hiển thị nhiều dạng vector (One-hot, BoW, TF-IDF, N-Gram, Word2Vec demo, FastText demo, BERT demo).
    *   Cho dữ liệu CSV: Người dùng chọn MỘT phương pháp (TF-IDF hoặc BoW được hỗ trợ chính) để vector hóa toàn bộ cột văn bản, chuẩn bị cho việc huấn luyện mô hình.

5.  **Huấn luyện mô hình (Train Tab):**
    *   Sử dụng dữ liệu CSV đã được vector hóa (bằng TF-IDF hoặc BoW).
    *   Huấn luyện các mô hình phân loại văn bản: Naive Bayes, Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting.
    *   Người dùng có thể tùy chỉnh tham số cho từng mô hình và tỷ lệ test size.
    *   Hiển thị kết quả: Accuracy, Classification Report, Confusion Matrix.
    *   Lưu model đã huấn luyện (tên do người dùng đặt) và vectorizer tương ứng.
    *   So sánh accuracy của các model đã lưu.
    *   Kiểm tra (dự đoán) trên văn bản mới bằng model đã lưu.

6.  **Gợi ý sản phẩm (Recommend Tab):**
    *   Sử dụng thuật toán SVD (Singular Value Decomposition).
    *   Người dùng tải lên một file CSV chứa dataset sản phẩm (cần cột mô tả sản phẩm và cột ID/tên sản phẩm).
    *   Hệ thống huấn luyện mô hình SVD dựa trên TF-IDF của mô tả sản phẩm.
    *   Người dùng nhập một đoạn mô tả/query, hệ thống gợi ý các sản phẩm tương tự nhất.

7.  **Chatbot (Chatbot Tab):**
    *   Có một chatbot dựa trên luật (KB_Bot) trả lời các câu hỏi cơ bản về NLP và ứng dụng.
    *   Bạn (Gemini) là chatbot thứ hai, cung cấp câu trả lời thông minh hơn dựa trên kiến thức chung và thông tin về project này được cung cấp trong prompt này.

Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng liên quan đến các chức năng của ứng dụng NLP WebApp Pro này hoặc các khái niệm NLP cơ bản được sử dụng trong ứng dụng. Hãy cố gắng trả lời một cách ngắn gọn, chính xác và hữu ích. Nếu câu hỏi không liên quan đến ứng dụng hoặc NLP, bạn có thể từ chối một cách lịch sự.
Luôn nhớ rằng bạn đang hoạt động trong khuôn khổ của ứng dụng NLP WebApp Pro.
---
"""
    # Kết hợp system prompt với câu hỏi của người dùng
    full_prompt_to_gemini = f"{system_prompt}\n\nCâu hỏi của người dùng: {user_message}\n\nCâu trả lời của bạn (với vai trò trợ lý AI của NLP WebApp Pro):"

    try:
        response = gemini_model_instance.generate_content(full_prompt_to_gemini)
        
        bot_reply = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                bot_reply += part.text
        
        if not bot_reply:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                bot_reply = f"[Gemini] Nội dung bị chặn vì: {response.prompt_feedback.block_reason.name}"
            elif response.candidates and response.candidates[0].finish_reason.name != "STOP":
                 bot_reply = f"[Gemini] Yêu cầu kết thúc vì: {response.candidates[0].finish_reason.name}"
            else:
                bot_reply = "[Gemini] Xin lỗi, tôi không có phản hồi cho câu này từ API."
        
        return jsonify({"reply": bot_reply})

    except Exception as e:
        # ... (phần xử lý lỗi giữ nguyên)
        print(f"Gemini API call error: {str(e)}")
        error_message = str(e).lower()
        if "api key not valid" in error_message or "permission denied" in error_message or "authentication" in error_message:
            return jsonify({"reply": "Lỗi xác thực API Key của Gemini hoặc không có quyền truy cập. Vui lòng kiểm tra lại."}), 403
        if "quota" in error_message:
            return jsonify({"reply": "Đã vượt quá hạn ngạch sử dụng API của Gemini."}), 429
        if "resource has been exhausted" in error_message: 
             return jsonify({"reply": "Yêu cầu tới Gemini quá nhanh, vui lòng thử lại sau giây lát."}), 429
        return jsonify({"reply": f"Đã có lỗi xảy ra khi giao tiếp với Gemini: {str(e)[:200]}..."}), 500
    
    
if __name__ == '__main__':
    # nltk.download('punkt', quiet=True)
    # nltk.download('wordnet', quiet=True)
    # nltk.download('stopwords', quiet=True)
    # try:
    #     nlp_spacy = spacy.load("en_core_web_sm")
    # except OSError:
    #     print("Downloading en_core_web_sm...")
    #     spacy.cli.download("en_core_web_sm")
    #     nlp_spacy = spacy.load("en_core_web_sm")
    app.run(host='0.0.0.0', port=5000, debug=True)