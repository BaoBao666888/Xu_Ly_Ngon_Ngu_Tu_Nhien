# NLP WebApp Pro 🌐

Ứng dụng web xử lý ngôn ngữ tự nhiên (NLP) đa năng: tăng cường dữ liệu, tiền xử lý, vector hóa, huấn luyện mô hình, gợi ý sản phẩm, chatbot AI.

## Tính năng

- Nhập liệu thủ công, tải file TXT/CSV, cào dữ liệu web
- Tăng cường dữ liệu (back-translation, synonym, v.v.)
- Tiền xử lý văn bản (tokenize, stopwords, viết thường, ...)
- Vector hóa: TF-IDF, BoW, Word2Vec, FastText, BERT, GPT-2, ELMo, GloVe, ...
- Huấn luyện mô hình: Naive Bayes, Logistic Regression, SVM, KNN, Tree, Random Forest, Gradient Boosting
- Lưu, so sánh, dự đoán với các model đã train
- Gợi ý sản phẩm dựa trên mô tả (SVD)
- Chatbot AI (mặc định & Gemini)

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Cài đặt Python 3.8+ và pip

### 3. Cài đặt các thư viện

```bash
pip install -r requirements.txt
```
Hoặc dùng file batch (Windows):

```bash
install_dependencies.bat
```

### 4. Tải model spaCy tiếng Anh

```bash
python -m spacy download en_core_web_sm
```
*(Nếu chưa tự động tải khi cài requirements)*

## Chạy ứng dụng

```bash
python app.py
```
Hoặc:
```bash
flask run
```

- Truy cập: http://localhost:5000

## Cấu trúc thư mục

- `app.py`: file chính Flask backend
- `templates/index.html`: giao diện web
- `static/js/webapp.js`: logic JS phía client
- `static/css/styles.css`: style giao diện

## Lưu ý

- Một số tính năng AI nâng cao (BERT, GPT-2, Gemini...) cần có GPU hoặc API key riêng.
- Để sử dụng Gemini hoặc OpenAI, cần thiết lập biến môi trường API_KEY tương ứng.

## Đóng góp

Pull request, issue, góp ý đều được hoan nghênh!

---

**Tác giả:** Nguyễn Quốc Bảo
