import math
from flask import Flask, jsonify, render_template, request, session
from flask_session import Session
from markupsafe import escape
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk
import os
import urllib.request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import spacy
from nltk.stem import PorterStemmer, SnowballStemmer
from transformers import MarianMTModel, MarianTokenizer, GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from nltk.util import ngrams
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import contractions
from allennlp.modules.elmo import Elmo, batch_to_ids
import datasets

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './sessions'
app.secret_key = 'your_secret_key'
Session(app)

# Tải model spaCy
nlp = spacy.load("en_core_web_sm")

def add_noise(text):
    aug = nac.RandomCharAug(action="insert")
    return aug.augment(text)[0]

def entity_replacement(text):
    doc = nlp(text)
    new_text = text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            new_text = new_text.replace(ent.text, "John")
    return new_text

def represent_data(text, representation_type):
    try:    
        # Lấy danh sách các lựa chọn từ form
        basic_methods = request.form.getlist("basic_representation")
        distributed_methods = request.form.getlist("distributed_representation")
        contextualized_methods = request.form.getlist("contextualized_representation")

        print("Selected Basic Methods:", basic_methods)
        print("Selected Distributed Methods:", distributed_methods)
        print("Selected Contextualized Methods:", contextualized_methods)

        representation_output = "<h3>Biểu diễn dữ liệu:</h3>"
        # Basic Vectorization Approaches
        if basic_methods:
            representation_output += "<h3>1. Basic Vectorization Approaches:</h3>"

            if "one_hot" in basic_methods:
                vocab = {word: i for i, word in enumerate(sorted(set(text)))}
                one_hot = np.zeros((len(text), len(vocab)))
                for i, word in enumerate(text):
                    one_hot[i, vocab[word]] = 1
                representation_output += f"<p><strong>One-hot:</strong><br> {escape(str(one_hot))}</p>"

            if "bow" in basic_methods:
                vectorizer = CountVectorizer()
                try:
                    bow = vectorizer.fit_transform([" ".join(text)])
                    representation_output += f"<p><strong>Bag of Words:</strong><br> {escape(str(bow.toarray()))}</p>"
                except ValueError as ve:
                    representation_output += f"<p><strong>Bag of Words:</strong> {escape(ve)} </p>"

            if "tfidf" in basic_methods:
                tfidf_vectorizer = TfidfVectorizer()
                try:
                    tfidf = tfidf_vectorizer.fit_transform([" ".join(text)])
                    representation_output += f"<p><strong>TF-IDF:</strong><br> {escape(str(tfidf.toarray()))}</p>"
                except ValueError as ve:
                    representation_output += f"<p><strong>TF-IDF:</strong> {escape(ve)}</p>"

            if "ngram" in basic_methods:
                n = 2
                bon_grams = list(ngrams(text, n))
                vectorizer_bon = CountVectorizer(analyzer="word", ngram_range=(n, n))
                try:
                    bow_bon = vectorizer_bon.fit_transform([" ".join(text)])
                    representation_output += f"<p><strong>Bag of N-grams (n={n}):</strong> {escape(str(bon_grams))}</p>"
                except ValueError as ve:
                    representation_output += f"<p><strong>Bag of N-grams (n={n}):</strong> {escape(ve)}</p>"

        # Distributed Representations
        if distributed_methods:
            representation_output += "<h3>2. Distributed Representations:</h3>"

            if "word2vec" in distributed_methods:
                model_w2v = Word2Vec([text], vector_size=100, window=5, min_count=1, workers=4)
                word_vectors_w2v = {word: model_w2v.wv[word].tolist() for word in text}
                representation_output += f"<p><strong>Word2Vec vectors:</strong> {escape(str(word_vectors_w2v))}</p>"

            if "fasttext" in distributed_methods:
                model_ft = FastText([text], vector_size=100, window=5, min_count=1, workers=4)
                word_vectors_ft = {word: model_ft.wv[word].tolist() for word in text}
                representation_output += f"<p><strong>FastText vectors:</strong> {escape(str(word_vectors_ft))}</p>"

            if "glove" in distributed_methods:
                representation_output += "<p><strong>GloVe chưa được tích hợp.</strong></p>"

        # Contextualized Representations
        if contextualized_methods:
            representation_output += "<h3>3. Contextualized Representation:</h3>"

            if "gpt" in contextualized_methods:
                tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
                model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
                inputs = tokenizer_gpt2(" ".join(text), return_tensors="pt")
                outputs = model_gpt2(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                representation_output += f"<p><strong>GPT-2 loss:</strong> {escape(str(loss))}</p>"

            if "elmo" in contextualized_methods:
                try:
                    # Load model ELMo
                    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                    elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
                    character_ids = batch_to_ids([text])
                    elmo_embeddings = elmo(character_ids)["elmo_representations"][0].detach().numpy()
                    representation_output += f"<p><strong>ELMo vectors:</strong> {escape(str(elmo_embeddings))}</p>"
                except Exception as e:
                    representation_output += f"<p><strong>ELMo vectors:</strong> {e}</p>"
            if "bert" in contextualized_methods:
                try:
                    # Load BERT model
                    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    bert_model = BertModel.from_pretrained("bert-base-uncased")
                    inputs = bert_tokenizer(" ".join(text), return_tensors="pt", padding=True, truncation=True)
                    outputs = bert_model(**inputs)
                    bert_embeddings = outputs.last_hidden_state.detach().numpy()
                    representation_output += f"<p><strong>BERT embeddings:</strong> {escape(str(bert_embeddings))}</p>"
                except Exception as e:
                    representation_output += f"<p><strong>BERT embeddings:</strong> {escape(e)}</p>"

        return representation_output

    except Exception as e:
        return f"<h3>Lỗi:</h3><p>{escape(str(e))}</p>"


def process_paragraph(text, selected_options):
    try:
        processing_result = text  # Giữ nguyên dữ liệu gốc
        processing_output = f"<h3>Xử lý câu mẫu:</h3>"
        processing_output += f"<p><strong>Văn bản gốc:</strong> {escape(text)}</p>"

        # Biến lưu kết quả tạm thời
        sentences, words, words_no_stop, pos_tags = [], [], [], []

        def create_table(data_list, title):
            """Tạo bảng nhiều cột tự động dựa theo độ dài chữ"""
            if not isinstance(data_list, list):
                return f"<p><strong>{title}:</strong> {escape(data_list)}</p>"

            num_items = len(data_list)
            max_length = max(len(word) for word in data_list) if data_list else 0

            # Xác định số cột phù hợp
            if num_items < 10:
                num_cols = 1
            elif num_items < 20:
                num_cols = 2
            elif max_length > 8:
                num_cols = 3
            else:
                num_cols = 4

            # Chia dữ liệu thành từng hàng với số cột xác định
            num_rows = math.ceil(num_items / num_cols)
            table_rows = []
            for i in range(num_rows):
                row = "<tr>"
                for j in range(num_cols):
                    index = i + j * num_rows  # Duyệt theo chiều dọc
                    if index < num_items:
                        row += f"<td>{escape(data_list[index])}</td>"
                    else:
                        row += "<td></td>"  # Ô trống nếu không đủ phần tử
                row += "</tr>"
                table_rows.append(row)

            return f"<p><strong>{title}:</strong></p><table border='1'>{''.join(table_rows)}</table>"


        for option_value in selected_options:
            if option_value == "sentence_tokenization":
                doc = nlp(processing_result)
                sentences = [sent.text for sent in doc.sents] if doc.sents else [text]
                processing_result = sentences  # Cập nhật dữ liệu
                processing_output += create_table(processing_result, "Chia thành các câu")

            elif option_value == "word_tokenization":
                words = []
                if isinstance(processing_result, list):
                    words = [token.text for sent in processing_result for token in nlp(sent)]
                else:
                    words = [token.text for token in nlp(processing_result)]
                processing_result = words
                processing_output += create_table(processing_result, "Chia thành các từ")

            elif option_value == "remove_stopwords":
                if isinstance(processing_result, list):
                    processing_result = [word.text for sent in processing_result for word in nlp(sent) if not word.is_stop]
                else:
                    processing_result = [word.text for word in nlp(processing_result) if not word.is_stop]
                    
                processing_output += create_table(processing_result, "Xóa stopwords")
            
            elif option_value == "rm_pun":
                if isinstance(processing_result, list):
                    processing_result = [word.text for sent in processing_result for word in nlp(sent) if not word.is_punct]
                else:
                    processing_result = [word.text for word in nlp(processing_result) if not word.is_punct]
                    
                processing_output += create_table(processing_result, "Xóa các loại dấu")

            elif option_value == "lowercasing":
                if isinstance(processing_result, list):
                    processing_result = [word.lower() for word in processing_result]
                else:
                    processing_result = processing_result.lower()
                processing_output += create_table(processing_result, "Viết thường")
            

            elif option_value == "fix_abbreviations":
                if isinstance(processing_result, list):
                    processing_result = [contractions.fix(word) for word in processing_result]
                else:
                    processing_result = contractions.fix(processing_result)
                processing_output += create_table(processing_result, "Sửa viết tắt")

            elif option_value == "pos_tagging":
                doc = nlp(" ".join(processing_result)) if isinstance(processing_result, list) else nlp(processing_result)
                pos_tags = [(token.text, token.pos_) for token in doc]
                pos_html = "".join(f"<tr><td>{escape(word)}</td><td>{escape(tag)}</td></tr>" for word, tag in pos_tags)
                processing_output += f"<p><strong>POS Tagging:</strong></p><table border='1'><tr><th>Từ</th><th>Loại từ</th></tr>{pos_html}</table>"

        return processing_result, processing_output

    except Exception as e:
        return f"<h3>Error:</h3><p>{escape(str(e))}</p>"



def translate(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def back_translation(text, src_lang="en", tgt_lang="fr"):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translated = translate(text, model, tokenizer)
    model_name_reverse = f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}'
    model_reverse = MarianMTModel.from_pretrained(model_name_reverse)
    tokenizer_reverse = MarianTokenizer.from_pretrained(model_name_reverse)
    return translate(translated, model_reverse, tokenizer_reverse)

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    error = ""
    collection_output = session.get('collection_output', '')
    augmentation_result = session.get('augmentation_result', '')
    processing_output = session.get('processing_output', '')
    representation_output = session.get('representation_output', '')

    if request.method == "POST":
        step = request.form.get("step")
        if step == "collection":
            input_type = request.form.get("input_type")
            if input_type == "manual":
                text = request.form.get("manual_text")
                if text:
                    session['collected_text'] = text
                    session['paragraphs'] = []
                    collection_output = f"<p>Đã thu thập dữ liệu từ nhập văn bản:</p><p>{escape(text)}</p>"
                else:
                    error = "Vui lòng nhập văn bản."
            elif input_type == "web":
                url = 'http://www.gutenberg.org/ebooks/1661.txt.utf-8'
                file_name = 'sherlock.txt'
                if not os.path.exists(file_name):
                    try:
                        with urllib.request.urlopen(url) as response:
                            with open(file_name, 'wb') as out_file:
                                out_file.write(response.read())
                    except Exception as e:
                        error = f"Lỗi tải sách: {str(e)}"
                try:
                    with open(file_name, "r", encoding="utf-8") as f:
                        text_data = f.read()
                        paragraphs = [p.strip() for p in text_data.split("\n\n") if p.strip()]
                    session['paragraphs'] = paragraphs
                    session['collected_text'] = ""
                    collection_output = "<p>Đã thu thập dữ liệu từ web. Vui lòng chọn đoạn.</p>"
                except Exception as e:
                    error = f"Lỗi xử lý sách: {str(e)}"
            session['collection_output'] = collection_output
        elif step == "select_paragraph":
            index = request.form.get("paragraph_index")
            try:
                idx = int(index)
                paragraphs = session.get('paragraphs', [])
                if idx < len(paragraphs):
                    session['collected_text'] = paragraphs[idx]
                    collection_output = f"<p>Đã chọn đoạn:</p><p>{escape(paragraphs[idx])}</p>"
                    session['collection_output'] = collection_output
                else:
                    error = "Chỉ số đoạn không hợp lệ."
            except Exception as e:
                error = f"Lỗi: {str(e)}"
        elif step == "augmentation":
            collected_text = session.get('collected_text', '')
            if not collected_text:
                error = "Chưa có dữ liệu thu thập. Vui lòng thực hiện Bước 1 trước."
            else:
                selected_options = request.form.getlist("augment_options")
                try:
                    augmented = collected_text
                    for option_value in selected_options:
                        if option_value == "back_translation":
                            augmented = back_translation(augmented)
                        elif option_value == "synonym_replacement":
                            aug = naw.SynonymAug(aug_src='wordnet')
                            augmented = aug.augment(augmented)[0]
                        elif option_value == "random_insertion":
                            aug = naw.RandomWordAug(action="substitute")
                            augmented = aug.augment(augmented)[0]
                        elif option_value == "random_swap":
                            aug = naw.RandomWordAug(action="swap")
                            augmented = aug.augment(augmented)[0]
                        elif option_value == "random_deletion":
                            aug = naw.RandomWordAug(action="delete")
                            augmented = aug.augment(augmented)[0]
                        elif option_value == "entity_replacement":
                            augmented = entity_replacement(augmented)
                        elif option_value == "add_noise":
                            augmented = add_noise(augmented)
                    session['augmented_text'] = augmented
                    augmentation_result = f"""
                    <p><strong>Văn bản gốc: </strong>{escape(collected_text)}</p>
                    <p>Kết quả tăng cường (theo thứ tự):</p><p>{escape(augmented)}</p>"""
                    session['augmentation_result'] = augmentation_result
                except Exception as e:
                    error = f"Lỗi trong quá trình tăng cường: {str(e)}"
        elif step == "processing":
            augmented_text = session.get('augmented_text', '')
            collected_text = session.get('collected_text', '')
            text_to_process = augmented_text if augmented_text else collected_text
            if not text_to_process:
                error = "Chưa có dữ liệu để xử lý."
            else:
                selected_options = request.form.getlist("preprocess_options")
                try:
                    processed_result, processing_output = process_paragraph(text_to_process, selected_options)
                    session['processed_result'] = processed_result
                    session['processing_output'] = processing_output
                except Exception as e:
                    error = f"Lỗi trong quá trình xử lý: {str(e)}"
        elif step == "representation":
            collected_text = session.get('collected_text', '')
            processed_result = session.get('processed_result', '')

            # Nếu processed_result là danh sách, nối lại thành chuỗi
            if isinstance(processed_result, list):
                text_to_process = ' '.join(processed_result)
            else:
                text_to_process = processed_result if processed_result else collected_text

            if not text_to_process:
                error = "Chưa có dữ liệu xử lý. Vui lòng thực hiện Bước 3 trước."
            else:
                representation_type = request.form.get('representation_type', 'basic')
                representation_output = represent_data(text_to_process, representation_type)
                session['representation_output'] = representation_output


    paragraphs = session.get('paragraphs', [])
    select_paragraph_form = ""
    if paragraphs:
        options_html = "<select name='paragraph_index'>"
        for i, para in enumerate(paragraphs):
            snippet = para[:50] + "..." if len(para) > 50 else para
            options_html += f"<option value='{i}'>Đoạn {i+1}: {escape(snippet)}</option>"
        options_html += "</select>"
        select_paragraph_form = f"""
        <div class="step" id="select_paragraph">
            <h3>Chọn đoạn cần xử lý từ dữ liệu sách</h3>
            <form method="POST">
                <input type="hidden" name="step" value="select_paragraph">
                {options_html}
                <br><br>
                <button type="submit">Chọn đoạn</button>
            </form>
        </div>
        """

    return render_template("index.html",
                           collection_output=collection_output,
                           select_paragraph_form=select_paragraph_form,
                           augmentation_result=augmentation_result,
                           processing_output=processing_output,
                           representation_output=representation_output,
                           error=error)


@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        dataset_name = data.get("dataset")
        model_type = data.get("model")

        # Lấy dataset từ Hugging Face (tên tương ứng)
        dataset_mapping = {
            "SST": "sst",
            "IMDb": "imdb",
            "Yelp": "yelp_review_full",
            "Amazon": "amazon_polarity",
            "TREC": "trec",
            "Yahoo": "yahoo_answers",
            "AG": "ag_news",
            "Sogou": "sogou_news",
            "DBPedia": "dbpedia_14"
        }
        
        if dataset_name not in dataset_mapping:
            return "Dataset không hợp lệ!", 400

        dataset = datasets.load_dataset(dataset_mapping[dataset_name], split='train[:5%]', trust_remote_code=True)  # Lấy 5% dữ liệu train
        check_dataset_structure(dataset)
        # return "check", 400 #tránh báo lỗi nhiều, ktra trước xem đã
        if dataset_name == "SST":
            x, y = dataset['sentence'], dataset['label']
            y = [1 if label >= 0.5 else 0 for label in y]  # Chuyển về nhãn 0 hoặc 1
        elif dataset_name == "Amazon":
            x, y = dataset['title'], dataset['label']
        elif dataset_name == "IMDb" or dataset_name == "Yelp" or dataset_name == "AG":
            x, y = dataset['text'], dataset['label']
        elif dataset_name == "TREC":
            x, y = dataset['text'], dataset['coarse_label']
        else:
            return "<h2>Hết dung lượng rồi!!!!!!!!!</h2>", 400

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)

        # Biến đổi văn bản thành vector TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        # Chọn mô hình
        if model_type == "knn":
            model = KNeighborsClassifier(n_neighbors=3)
        elif model_type == "decision_tree":
            model = DecisionTreeClassifier(max_depth=5)
        else:
            return "Mô hình không hợp lệ!", 400

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        report = metrics.classification_report(y_test, pred)

        result = {
            "dataset": dataset_name,
            "model": model_type,
            "accuracy": f"{accuracy:.4f}",
            "report": report
        }
        if "results" not in session:
            session["results"] = []
        session["results"].append(result)
        session.modified = True
        
        return jsonify(result)
    except Exception as e:
        return str(e), 400

@app.route('/results', methods=['GET'])
def get_results():
    """Lấy danh sách kết quả đã lưu"""
    return jsonify(session.get("results", []))

@app.route('/delete_result', methods=['POST'])
def delete_result():
    try:
        data = request.json
        index = data.get("index")

        if "results" not in session or index is None or index >= len(session["results"]):
            return "Index không hợp lệ!", 400

        session["results"].pop(index)  # Xóa kết quả
        session.modified = True  # Cập nhật session

        return jsonify({"results": session["results"]})  
    except Exception as e:
        return str(e), 400


@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Xóa toàn bộ kết quả"""
    session.pop("results", None)
    return jsonify({"message": "Đã xóa tất cả kết quả."})


def check_dataset_structure(dataset):
    """
    Kiểm tra cấu trúc của dataset, in ra danh sách cột và 5 dòng đầu tiên.
    """
    if isinstance(dataset, datasets.DatasetDict):  # Trường hợp nhiều tập dữ liệu (train, test, validation)
        for split_name, split_data in dataset.items():
            print(f"\n=== Checking '{split_name}' dataset ===")
            print("Columns:", split_data.column_names)
            print("Sample Data:")
            for i in range(min(5, len(split_data))):
                print(split_data[i])
    elif isinstance(dataset, datasets.Dataset):  # Nếu chỉ có một tập dữ liệu
        print("\n=== Checking dataset ===")
        print("Columns:", dataset.column_names)
        print("Sample Data:")
        for i in range(min(5, len(dataset))):
            print(dataset[i])
    else:
        print("Unknown dataset type!")


if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    app.run(debug=True, host="0.0.0.0", port=5555)