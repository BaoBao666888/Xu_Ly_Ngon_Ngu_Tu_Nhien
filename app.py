from flask import Flask, render_template, request, session
from flask_session import Session
from markupsafe import escape
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk
import os
import urllib.request
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer, SnowballStemmer
from transformers import MarianMTModel, MarianTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.util import ngrams
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

def process_paragraph(text):
    try:
        doc_spacy = nlp(text)
        sentences = list(doc_spacy.sents)
        sample_sentence = sentences[0].text if sentences else text

        # split_text, spacy_tokenize, remove_stopwords, remove_punctuation, stemming, lemmatization, pos_tagging
        doc = nlp(text)
        words = [token.text for token in doc]  # spacy_tokenize
        words_no_stop = [token.text for token in doc if token.text.lower() not in STOP_WORDS and not token.is_punct and not token.is_space]  # remove_stopwords, remove_punctuation
        lemmatized_words = [token.lemma_ for token in nlp(" ".join(words_no_stop))]  # lemmatization
        porter_stemmer = PorterStemmer()
        stemmed_words_porter = [porter_stemmer.stem(w) for w in lemmatized_words]  # stemming
        snowball_stemmer = SnowballStemmer("english")
        stemmed_words_snowball = [snowball_stemmer.stem(w) for w in lemmatized_words]  # stemming
        pos_tags = [(token.text, token.pos_) for token in doc]  # pos_tagging

        # one_hot_encoding, bag_of_words
        vocab = {word: i for i, word in enumerate(sorted(set(words_no_stop)))}
        one_hot = np.zeros((len(words_no_stop), len(vocab)))
        for i, word in enumerate(words_no_stop):
            one_hot[i, vocab[word]] = 1
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform([sample_sentence])

        # boN, fasttext, glove, gpt2, tfidf
        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform([sample_sentence])
        n = 2
        bon_grams = list(ngrams(words_no_stop, n))
        vectorizer_bon = CountVectorizer(analyzer='word', ngram_range=(n, n))
        bow_bon = vectorizer_bon.fit_transform([" ".join(words_no_stop)])
        model_ft = FastText([words_no_stop], vector_size=100, window=5, min_count=1, workers=4)
        word_vectors_ft = {word: model_ft.wv[word] for word in words_no_stop}
        # glove_input_file = 'glove/glove.6B.100d.txt'
        # word2vec_output_file = 'glove/glove.6B.100d.word2vec.txt'
        # if not os.path.exists(word2vec_output_file):
        #     glove2word2vec(glove_input_file, word2vec_output_file)
        # model_glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        # word_vectors_glove = {word: model_glove[word] for word in words_no_stop if word in model_glove}
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
        model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        inputs = tokenizer_gpt2(sample_sentence, return_tensors="pt")
        outputs = model_gpt2(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        #test naive_bayes, maximizing_likelihood
        train_data = ["This is a positive sentence.", "Another positive example.", "This is negative.", "Negative example."]
        labels = [1, 1, 0, 0]
        X_train = tfidf_vectorizer.fit_transform(train_data)
        X_test = tfidf_vectorizer.transform([sample_sentence])
        model_nb = MultinomialNB()
        model_nb.fit(X_train, labels)
        prediction_nb = model_nb.predict(X_test)[0]
        probability_nb = model_nb.predict_proba(X_test)[0]
        model_lr = LogisticRegression()
        model_lr.fit(X_train, labels)
        prediction_lr = model_lr.predict(X_test)[0]
        probability_lr = model_lr.predict_proba(X_test)[0]
        model_svm = SVC(probability=True)
        model_svm.fit(X_train, labels)
        prediction_svm = model_svm.predict(X_test)[0]
        probability_svm = model_svm.predict_proba(X_test)[0]

        # Kết quả
        processing_result = f"<h3>Xử lý câu mẫu:</h3>"
        processing_result += f"<p><strong>Câu gốc:</strong> {escape(sample_sentence)}</p>"
        processing_result += f"<p><strong>Token:</strong> {escape(str(words_no_stop))}</p>"
        processing_result += f"<p><strong>Lemmatized:</strong> {escape(str(lemmatized_words))}</p>"
        processing_result += f"<p><strong>Stem (Porter):</strong> {escape(str(stemmed_words_porter))}</p>"
        processing_result += f"<p><strong>Stem (Snowball):</strong> {escape(str(stemmed_words_snowball))}</p>"
        processing_result += f"<p><strong>POS Tagging:</strong> {escape(str(pos_tags))}</p>"
        processing_result += f"<p><strong>One-hot:</strong><br> {escape(str(one_hot))}</p>"
        processing_result += f"<p><strong>Bag of Words:</strong><br> {escape(str(bow.toarray()))}</p>"
        processing_result += f"<p><strong>TF-IDF:</strong><br> {escape(str(tfidf.toarray()))}</p>"
        processing_result += f"<p><strong>Bag of N-grams (n={n}):</strong> {escape(str(bon_grams))}</p>"
        processing_result += f"<p><strong>FastText vectors:</strong> {escape(str(word_vectors_ft))}</p>"
        # processing_result += f"<p><strong>GloVe vectors:</strong> {escape(str(word_vectors_glove))}</p>"
        processing_result += f"<p><strong>GPT-2 loss:</strong> {escape(str(loss.item()))}</p>"
        processing_result += "<h3>Text Classification:</h3>"
        processing_result += f"<p><strong>Naive Bayes:</strong> Dự đoán: {prediction_nb}, Xác suất: {probability_nb}</p>"
        processing_result += f"<p><strong>Logistic Regression:</strong> Dự đoán: {prediction_lr}, Xác suất: {probability_lr}</p>"
        processing_result += f"<p><strong>SVM:</strong> Dự đoán: {prediction_svm}, Xác suất: {probability_svm}</p>"
        return processing_result
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
                    augmentation_result = f"<p>Kết quả tăng cường (theo thứ tự):</p><p>{escape(augmented)}</p>"
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
                try:
                    processing_output = process_paragraph(text_to_process)
                    session['processed_result'] = processing_output
                    session['processing_output'] = processing_output
                except Exception as e:
                    error = f"Lỗi trong quá trình xử lý: {str(e)}"
        elif step == "representation":
            processed_result = session.get('processed_result', '')
            if not processed_result:
                error = "Chưa có dữ liệu xử lý. Vui lòng thực hiện Bước 3 trước."
            else:
                pass  # Chưa triển khai

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
                           error=error)

if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    app.run(debug=True, host="0.0.0.0", port=5555)