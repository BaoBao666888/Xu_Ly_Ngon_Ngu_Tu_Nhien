#Nguyễn Quốc Bảo - MSSV: 22110109
from flask import Flask, render_template, request, jsonify
import nlpaug.augmenter.word as naw
import difflib
import nltk
from nlpaug.augmenter.word import SynonymAug
from markupsafe import escape
import os

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        text = escape(text)  # Escape HTML đặc biệt
        selected_options = request.form.getlist("options")
        augment_method = request.form.get("augment_method")

        if text:
            text = str(text)
            results = []
            option_map = {
                "back_translation": "Dịch ngược (EN-VI-EN)",
                "synonym_replacement": "Thay thế từ đồng nghĩa",
                "random_insertion": "Thêm ngẫu nhiên một số từ",
                "random_swap": "Đổi chỗ ngẫu nhiên một số từ",
                "random_deletion": "Xóa ngẫu nhiên một số từ",
            }

            output_file = "text.txt"
            # Xóa file nếu đã tồn tại
            if os.path.exists(output_file):
                os.remove(output_file)

            with open(output_file, "a", encoding="utf-8") as f:
                f.write("Văn bản gốc:\n")
                f.write(text + "\n\n")

            for i, option_value in enumerate(selected_options):
                option_name = option_map.get(option_value)
                if not option_name:  
                    continue
                if option_name == "Dịch ngược (EN-VI-EN)":
                    aug = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-vi', to_model_name='Helsinki-NLP/opus-mt-vi-en')
                elif option_name == "Thay thế từ đồng nghĩa":
                    aug = naw.SynonymAug(aug_src='wordnet')
                elif option_name == "Thêm ngẫu nhiên một số từ":
                    aug = naw.RandomWordAug(action="substitute")
                elif option_name == "Đổi chỗ ngẫu nhiên một số từ":
                    aug = naw.RandomWordAug(action="swap")
                elif option_name == "Xóa ngẫu nhiên một số từ":
                    aug = naw.RandomWordAug(action="delete")
                else:
                    continue

                if augment_method == "Tăng cường theo input":
                    augmented_text = aug.augment(text)
                elif augment_method == "Tăng cường theo thứ tự" and i > 0:
                    augmented_text = aug.augment(augmented_text)
                else:  # Trường hợp đầu tiên trong "Tăng cường theo thứ tự"
                    augmented_text = aug.augment(text)

                # Chuyển đổi kết quả về chuỗi nếu cần
                if isinstance(augmented_text, list):
                    augmented_text = " ".join(augmented_text)

                # Tô màu từ khác biệt:
                diff = difflib.ndiff(text.split(), augmented_text.split())
                highlighted_text = ""
                for word in diff:
                    if word.startswith("+ "):
                        highlighted_text += f"<span style='color:red;'>{word[2:]} </span>"
                    elif word.startswith("- "):
                        pass
                    else:
                        highlighted_text += f"{word[2:]} "

                results.append({
                    "option": f"{i+1}. {option_name}",
                    "augmented_text": highlighted_text
                })
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"Kết quả:\n")
                    f.write(f"{i+1}. {option_name}:\n")
                    f.write(augmented_text + "\n\n")

            return render_template("index.html", results=results, original_text=text)
        else:
            return render_template("index.html", error="Vui lòng nhập văn bản.")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
