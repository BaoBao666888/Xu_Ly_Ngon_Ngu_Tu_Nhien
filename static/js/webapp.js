// Global state variables (use sparingly, prefer passing data or using session)
window.currentCSVSample = null;
window.csvTextCol = null;
window.csvLabelCol = null;
window.lastTrainedModelIdForSaving = null; // To enable "Save Model" button

function showTab(tabId, clickedButton) {
    document.querySelectorAll('.tab-content').forEach(div => div.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    const tabToShow = document.getElementById(tabId);
    if (tabToShow) {
        tabToShow.style.display = 'block';
    }
    if (clickedButton) {
        clickedButton.classList.add('active');
    }

    // Special actions when certain tabs are shown
    if (tabId === 'train') {
        loadSavedModelsForTesting(); // Populate dropdown in "Test Model" section
        updateModelParams(); // Initialize params for default model
    }
}

function withLoading(loaderId, asyncCallback) {
    const loader = document.getElementById(loaderId);
    if (loader) loader.style.display = 'inline-block';
    
    return asyncCallback()
        .catch(error => {
            console.error("Error in async operation:", error);
            alert("Đã có lỗi xảy ra: " + error.message);
        })
        .finally(() => {
            if (loader) loader.style.display = 'none';
        });
}

// ==== INPUT TAB ====
function loadTxtFile(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('manualText').value = e.target.result;
            alert("File TXT đã được tải vào ô văn bản.");
        };
        reader.readAsText(file);
    }
}

function doScrape() {
    const url = document.getElementById("scrape-url").value;
    if (!url) {
        alert("Vui lòng nhập URL.");
        return Promise.resolve();
    }
    return fetch("/scrape_web", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi cào web: " + data.error);
        } else if (data.scraped_text) {
            const manualTextArea = document.getElementById("manualText");
            manualTextArea.value += (manualTextArea.value ? "\n\n---SCRAPED---\n" : "") + data.scraped_text;
            alert("Đã cào xong văn bản và thêm vào ô 'Văn bản thủ công'.");
        } else {
            alert("Không cào được nội dung nào hoặc nội dung không đáng kể.");
        }
    });
}

function uploadCSV() {
    const fileInput = document.getElementById("csvInput");
    if (!fileInput.files.length) {
        alert("Vui lòng chọn một file CSV.");
        return Promise.resolve();
    }
    const formData = new FormData();
    formData.append("csv_file", fileInput.files[0]);

    return fetch("/upload_csv_input", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert("Lỗi tải CSV: " + data.error);
                return;
            }
            const selectText = document.getElementById("text-column");
            const selectLabel = document.getElementById("label-column");
            selectText.innerHTML = data.columns.map(col => `<option value="${col}">${col}</option>`).join('');
            selectLabel.innerHTML = data.columns.map(col => `<option value="${col}">${col}</option>`).join('');
            document.getElementById("column-selection").style.display = "block";

            // Display CSV sample
            window.currentCSVSample = data.sample; // Store for potential future use
            const sampleDiv = document.getElementById("csv-sample-display");
            if (data.sample && data.sample.length > 0) {
                let html = `<p><strong>3 dòng đầu tiên từ CSV:</strong></p><div class="result-table-container"><table>`;
                html += `<thead><tr>${Object.keys(data.sample[0]).map(key => `<th>${key}</th>`).join('')}</tr></thead><tbody>`;
                data.sample.forEach(row => {
                    html += `<tr>${Object.values(row).map(val => `<td>${String(val).substring(0,50)}${String(val).length > 50 ? '...' : ''}</td>`).join('')}</tr>`;
                });
                html += `</tbody></table></div>`;
                sampleDiv.innerHTML = html;
            } else {
                sampleDiv.innerHTML = "<p><em>Không có dữ liệu mẫu để hiển thị hoặc CSV rỗng.</em></p>";
            }
        });
}

function selectCSVColumns() {
    const text_col = document.getElementById("text-column").value;
    const label_col = document.getElementById("label-column").value;
    
    if (text_col === label_col) {
        alert("Cột văn bản và cột nhãn không được trùng nhau!");
        return;
    }

    return fetch("/select_csv_cols", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text_col, label_col })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi chọn cột: " + data.error);
        } else {
            window.csvTextCol = text_col; // Store globally for this session
            window.csvLabelCol = label_col;
            alert(`Đã chọn: Văn bản = ${text_col}, Nhãn = ${label_col}. Bạn có thể qua các tab khác.`);
            document.getElementById("column-selection").style.display = "none";
             // Automatically switch to Augment tab as a suggestion
            const augmentButton = Array.from(document.querySelectorAll('.tab-btn')).find(btn => btn.textContent.includes('Tăng cường'));
            if (augmentButton) showTab('augment', augmentButton);
        }
    });
}

// ==== AUGMENTATION ====
function applyAugmentation(isCsvCall) {
    const text = document.getElementById("manualText").value;
    const options = Array.from(document.querySelectorAll("#augment input[type='checkbox']:checked")).map(cb => cb.value);

    if (!options.length) {
        alert("Vui lòng chọn ít nhất một phương pháp tăng cường.");
        return Promise.resolve();
    }

    let payload = { options };
    if (isCsvCall) {
        if (!window.csvTextCol) {
            alert("Vui lòng tải CSV và chọn cột văn bản trước khi tăng cường CSV.");
            return Promise.resolve();
        }
        payload.is_csv = true; // Backend will use session data for CSV path and text_col
    } else {
        if (!text.trim()) {
            alert("Vui lòng nhập văn bản ở tab 'Nhập liệu' để tăng cường thủ công.");
            return Promise.resolve();
        }
        payload.text = text;
        payload.is_csv = false;
    }
    
    return fetch("/augment_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        const resultDiv = document.getElementById("augmented-result");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color:red;">Lỗi: ${data.error}</p>`;
        } else if (data.results) {
            if (data.is_csv) {
                let html = "<table><tr><th>Văn bản gốc (mẫu)</th><th>Văn bản tăng cường (mẫu)</th></tr>";
                data.results.forEach(pair => {
                    html += `<tr><td>${escapeHtml(pair.original)}</td><td>${escapeHtml(pair.augmented)}</td></tr>`;
                });
                html += "</table>";
                resultDiv.innerHTML = html;
            } else { // Manual
                resultDiv.innerHTML = `<p><strong>Kết quả tăng cường:</strong></p><pre>${escapeHtml(data.results[0].augmented)}</pre>`;
            }
        }
    });
}
function applyAugmentManual() { return applyAugmentation(false); }
function applyAugmentCSV() { return applyAugmentation(true); }


// ==== PREPROCESSING ====
function applyPreprocessing(isCsvCall) {
    const text = document.getElementById("manualText").value;
    const options = Array.from(document.querySelectorAll("#preprocess input[type='checkbox']:checked")).map(cb => cb.value);

    if (!options.length) {
        alert("Vui lòng chọn ít nhất một phương pháp tiền xử lý.");
        return Promise.resolve();
    }
    
    let payload = { options };
    if (isCsvCall) {
        if (!window.csvTextCol) {
            alert("Vui lòng tải CSV và chọn cột văn bản trước khi tiền xử lý CSV.");
            return Promise.resolve();
        }
        payload.is_csv = true;
    } else {
        if (!text.trim()) {
            alert("Vui lòng nhập văn bản ở tab 'Nhập liệu' để tiền xử lý thủ công.");
            return Promise.resolve();
        }
        payload.text = text;
        payload.is_csv = false;
    }

    return fetch("/preprocess_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        const resultContainer = document.getElementById("preprocess-result-container");
        const resultPre = document.getElementById("preprocess-result");
        if (data.error) {
            resultPre.innerText = "Lỗi: " + data.error;
            resultContainer.innerHTML = `<pre>${resultPre.innerText}</pre>`; // Ensure it's wrapped if pre is cleared
        } else if (data.results) {
             if (data.is_csv) {
                let html = "<table><tr><th>Văn bản gốc (mẫu)</th><th>Văn bản đã xử lý (mẫu)</th></tr>";
                data.results.forEach(pair => {
                    html += `<tr><td>${escapeHtml(pair.original)}</td><td>${escapeHtml(pair.processed)}</td></tr>`;
                });
                html += "</table>";
                resultContainer.innerHTML = html; // Replace pre with table
            } else { // Manual
                 resultPre.innerText = JSON.stringify(data.results[0].processed, null, 2);
                 resultContainer.innerHTML = ''; // Clear previous table if any
                 resultContainer.appendChild(resultPre); // Add pre back
            }
        }
    });
}
function applyPreprocessManual() { return applyPreprocessing(false); }
function applyPreprocessCSV() { return applyPreprocessing(true); }


// ==== VECTORIZATION ====
// webapp.js
function applyVectorizeManual() {
    const text = document.getElementById("manualText").value.trim();
    if (!text) {
        alert("Nhập văn bản ở tab 'Nhập liệu' để vector hoá thủ công.");
        return Promise.resolve();
    }
    // Lấy giá trị của phương thức đã chọn
    const selectedMethod = document.getElementById("vector-method").value;

    // Kiểm tra xem selectedMethod có giá trị không (select thường có giá trị mặc định)
    if (!selectedMethod) { // Thêm kiểm tra này cho chắc chắn
        alert("Vui lòng chọn một phương pháp vector hoá.");
        return Promise.resolve();
    }

    return fetch("/vectorize_manual_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Backend mong đợi một mảng 'methods', nên ta bọc selectedMethod trong mảng
        body: JSON.stringify({ text, methods: [selectedMethod] })
    })
    .then(res => res.json())
    .then(data => {
        const resultDiv = document.getElementById("vector-result-manual");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color:red;">Lỗi: ${data.error}</p>`;
            return;
        }
        let html = "<h4>Kết quả Vector Hoá Thủ Công:</h4>";
        // data.vectors bây giờ sẽ chỉ chứa một key là selectedMethod
        for (const [method, vectorData] of Object.entries(data.vectors)) {
            html += `<h5>${method.toUpperCase()}</h5>`;
            if (vectorData.error) {
                html += `<p style="color:orange;">Lỗi ${method}: ${vectorData.error}</p>`;
            } else if (method === "one_hot" || method === "word2vec" || method === "fasttext") {
                html += renderKeyValueTable(vectorData);
            } else if (method === "bow" || method === "tfidf") {
                html += `<p><strong>Features:</strong> ${vectorData.features ? vectorData.features.slice(0,10).join(', ') + (vectorData.features.length > 10 ? '...' : '') : 'N/A'}</p>`;
                html += renderVectorArray(vectorData.vector, 10);
            } else if (method === "ngram") {
                html += `<p>${vectorData.join ? vectorData.join(', ') : "Không có N-gram nào."}</p>`; // Thêm kiểm tra .join
            } else if (method === "bert") { // Giả sử BERT demo trả về cấu trúc này
                html += renderVectorArray(vectorData.sentence_vector_CLS_pooling_demo, 10);
            } else {
                 html += `<pre>${escapeHtml(JSON.stringify(vectorData, null, 2))}</pre>`;
            }
        }
        resultDiv.innerHTML = html;
    });
}

function applyVectorizeCSV() {
    if (!window.csvTextCol || !window.csvLabelCol) {
        alert("Chưa chọn cột Text/Label từ CSV đã tải lên ở tab 'Nhập liệu'!");
        return Promise.resolve();
    }
    const selectedMethod = document.getElementById("vector-method").value;    
    if (!selectedMethod) { // Kiểm tra nếu không có gì được chọn (dù select thường có giá trị mặc định)
        alert("Vui lòng chọn một phương pháp vector hoá.");
        return Promise.resolve();
    }

    return fetch("/vectorize_csv_data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ methods: [selectedMethod] })
    })
    .then(res => {
        const contentType = res.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            return res.json();
        } else {
            return res.text().then(text => { // Lấy text nếu không phải JSON
                throw new Error("Server response was not JSON. Received: " + text.substring(0, 200) + "...");
            });
        }
    })
    .then(data => {
        if (data.error) {
            alert("Lỗi Vector hoá CSV: " + data.error);
            document.getElementById("csv-vector-info").innerHTML = `<p style="color:red;">${data.error}</p>`;
            document.getElementById("csv-vector-samples").innerHTML = '';
            return;
        }
        document.getElementById("csv-vector-info").innerHTML =
            `<p><b>${data.message}</b><br>
             Số dòng xử lý: ${data.num_samples_vectorized}.<br>
             Chiều vector cho training (${data.training_vectorizer}): ${data.dimension_for_training}.</p>`;
        
        let htmlSamples = "<h4>Mẫu Vector Hoá CSV (hiển thị 3 dòng đầu, tối đa 10 chiều):</h4>";
        if (data.display_samples && Object.keys(data.display_samples).length > 0) {
            for (const [method, samples] of Object.entries(data.display_samples)) {
                htmlSamples += `<h5>${method.toUpperCase()}</h5>`;
                if (samples && samples.length > 0 && !samples[0].error) { // Check if first sample is not an error string
                    htmlSamples += "<table><thead><tr><th>Văn bản gốc</th><th>Vector (mẫu)</th></tr></thead><tbody>";
                    samples.forEach((sampleVector, index) => {
                        const originalText = data.sample_original_texts[index] || "N/A";
                        htmlSamples += `<tr><td>${escapeHtml(originalText.substring(0,50))}...</td><td>`;
                        if (method === "one_hot" && typeof sampleVector === 'object') { // one_hot is dict {token: vector}
                             htmlSamples += Object.entries(sampleVector).map(([token, vec]) => `${token}: [${vec.join(', ')}]`).join('<br>');
                        } else if (Array.isArray(sampleVector)) { // BOW, TFIDF give array
                            htmlSamples += `[${sampleVector.map(v => v.toFixed ? v.toFixed(3) : v).join(', ')}]`;
                        } else {
                            htmlSamples += String(sampleVector); // Fallback for other structures
                        }
                        htmlSamples += `</td></tr>`;
                    });
                    htmlSamples += "</tbody></table>";
                } else {
                     htmlSamples += `<p><em>Không có mẫu hoặc có lỗi cho phương pháp ${method}. ${samples && samples[0] ? samples[0].error : ''}</em></p>`;
                }
            }
        } else {
            htmlSamples += "<p><em>Không có mẫu vector nào được tạo.</em></p>";
        }
        document.getElementById("csv-vector-samples").innerHTML = htmlSamples;
         // Suggest moving to Train tab
        alert("Vector hóa CSV hoàn tất! Phương pháp đầu tiên bạn chọn (" + data.training_vectorizer + ") sẽ được dùng để Train. Bạn có thể qua tab Train.");
    });
}


// ==== TRAINING TAB ====
function updateModelParams() {
    const modelType = document.getElementById("model-type").value;
    const paramsDiv = document.getElementById("model-params");
    let html = "<h4>Thông số Model:</h4>";
    if (modelType === "logistic") {
        html += '<label>C (Inverse regularization strength): <input type="number" name="C" value="1.0" step="0.1"></label><br>';
        html += '<label>Penalty (l1, l2, elasticnet, None): <input type="text" name="penalty" value="l2"></label><br>';
        html += '<label>Solver (liblinear, saga...): <input type="text" name="solver" value="liblinear"></label><br>';
        html += '<label>Max Iterations: <input type="number" name="max_iter" value="1000" step="100"></label><br>';
    } else if (modelType === "svm") {
        html += '<label>C: <input type="number" name="C" value="1.0" step="0.1"></label><br>';
        html += '<label>Kernel (linear, rbf, poly, sigmoid): <input type="text" name="kernel" value="rbf"></label><br>';
        html += '<label>Gamma (float, "scale", "auto"): <input type="text" name="gamma" value="scale"></label><br>';
        html += '<label>Degree (for poly kernel): <input type="number" name="degree" value="3" min="1"></label><br>';
    } else if (modelType === "knn") {
        html += '<label>Number of Neighbors (n_neighbors): <input type="number" name="n_neighbors" value="5" min="1"></label><br>';
        html += '<label>Weights (uniform, distance): <input type="text" name="weights" value="uniform"></label><br>';
    } else if (modelType === "tree") {
        html += '<label>Criterion (gini, entropy): <input type="text" name="criterion" value="gini"></label><br>';
        // Cho phép nhập "None" dưới dạng string cho max_depth
        html += '<label>Max Depth (integer or "None"): <input type="text" name="max_depth" value="None"></label><br>';
    } else if (modelType === "rf") { // Random Forest
        html += '<label>Number of Trees (n_estimators): <input type="number" name="n_estimators" value="100" min="10"></label><br>';
        html += '<label>Criterion (gini, entropy): <input type="text" name="criterion" value="gini"></label><br>';
        html += '<label>Max Depth (integer or "None"): <input type="text" name="max_depth" value="None"></label><br>';
    } else if (modelType === "gb") { // Gradient Boosting
        html += '<label>Number of Estimators (n_estimators): <input type="number" name="n_estimators" value="100" min="10"></label><br>';
        html += '<label>Learning Rate: <input type="number" name="learning_rate" value="0.1" step="0.01" min="0.001"></label><br>';
        html += '<label>Max Depth: <input type="number" name="max_depth" value="3" min="1"></label><br>';
    } else { // Naive Bayes
        html += "<p><em>Naive Bayes (MultinomialNB) không có nhiều tham số tùy chỉnh phổ biến ở đây.</em></p>";
    }
    paramsDiv.innerHTML = html;
}

function trainFromCSV() {
    const model_type = document.getElementById("model-type").value;
    const test_size = document.getElementById("test-size").value;
    const params = {};
    document.querySelectorAll("#model-params input").forEach(input => {
        // Handle "None" string for parameters that can be None
        let value = input.value.trim();
        if (value.toLowerCase() === "none" || value === "") {
            // For some params, empty might mean default or None. Backend should handle.
            // For max_depth in Decision Tree, empty or "None" should translate to None type.
            if (input.name === 'max_depth' && (value.toLowerCase() === "none" || value === "")) {
                 params[input.name] = null; // Send null to backend
            } else if (value !== "") { // Only add if not empty (unless specific handling like above)
                params[input.name] = value;
            }
        } else if (!isNaN(parseFloat(value)) && input.type === "number") {
             params[input.name] = parseFloat(value); // Ensure numbers are numbers
        } else {
            params[input.name] = value;
        }
    });

    return fetch("/train_model_from_csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type, test_size, params })
    })
    .then(res => res.json())
    .then(data => {
        const resultTextDiv = document.getElementById("train-result-text");
        const chartsDiv = document.getElementById("train-charts");
        chartsDiv.innerHTML = ''; // Clear previous charts

        if (data.error) {
            resultTextDiv.innerText = "Lỗi huấn luyện: " + data.error;
            window.lastTrainedModelIdForSaving = null; // Disable save button
        } else {
            let reportHtml = `<strong>${data.message}</strong><br>Accuracy: ${data.accuracy.toFixed(4)}<br><br>`;
            reportHtml += "<strong>Classification Report:</strong><br>";
            reportHtml += "<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>";
            for (const className in data.report) {
                if (className !== "accuracy" && typeof data.report[className] === 'object') { // Iterate over class reports
                    const metrics = data.report[className];
                    reportHtml += `<tr>
                        <td>${className}</td>
                        <td>${metrics.precision.toFixed(3)}</td>
                        <td>${metrics.recall.toFixed(3)}</td>
                        <td>${metrics['f1-score'].toFixed(3)}</td>
                        <td>${metrics.support}</td>
                    </tr>`;
                } else if (className === "macro avg" || className === "weighted avg") { // Averages
                     const metrics = data.report[className];
                     reportHtml += `<tr>
                        <td><strong>${className.toUpperCase()}</strong></td>
                        <td>${metrics.precision.toFixed(3)}</td>
                        <td>${metrics.recall.toFixed(3)}</td>
                        <td>${metrics['f1-score'].toFixed(3)}</td>
                        <td>${metrics.support}</td>
                    </tr>`;
                }
            }
            reportHtml += "</tbody></table>";
            resultTextDiv.innerHTML = reportHtml; // Use innerHTML for table

            if (data.confusion_matrix_url) {
                chartsDiv.innerHTML += `<h4>Confusion Matrix:</h4><img src="${data.confusion_matrix_url}" alt="Confusion Matrix">`;
            }
            window.lastTrainedModelIdForSaving = data.model_id_for_saving; // Enable "Save Model"
            alert("Huấn luyện hoàn tất! Kết quả và biểu đồ đã được hiển thị.");
        }
    });
}

function saveTrainedModel() {
    const userModelName = document.getElementById("save-model-name").value.trim();
    if (!userModelName) {
        alert("Vui lòng đặt tên cho model trước khi lưu.");
        return Promise.resolve();
    }
    if (!window.lastTrainedModelIdForSaving) { // This ID is set by trainFromCSV on success
        alert("Chưa có model nào được huấn luyện thành công trong phiên này để lưu, hoặc model đang được lưu có ID khác.");
        return Promise.resolve();
    }

    return fetch("/save_trained_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_model_name: userModelName })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi lưu model: " + data.error);
        } else {
            alert(data.message);
            loadSavedModelsForTesting(); // Refresh dropdown list
            document.getElementById("save-model-name").value = ""; // Clear name input
            // window.lastTrainedModelIdForSaving = null; // Optionally clear to prevent re-saving same ID
        }
    });
}

function loadSavedModelsForTesting() {
    return fetch("/get_saved_models_list")
    .then(res => res.json())
    .then(data => {
        const select = document.getElementById("select-test-model");
        if (data.saved_models && data.saved_models.length > 0) {
            select.innerHTML = data.saved_models
                .map(m => `<option value="${m.disk_name}">${m.user_name} (Acc: ${typeof m.accuracy === 'number' ? m.accuracy.toFixed(3) : m.accuracy}, Type: ${m.type})</option>`)
                .join('');
        } else {
            select.innerHTML = "<option value=''>-- Chưa có model nào được lưu --</option>";
        }
    });
}

function predictWithSelectedModel() {
    const disk_model_name = document.getElementById("select-test-model").value;
    const text = document.getElementById("test-input-text").value.trim();

    if (!disk_model_name) {
        alert("Vui lòng chọn một model đã lưu.");
        return Promise.resolve();
    }
    if (!text) {
        alert("Vui lòng nhập văn bản để dự đoán.");
        return Promise.resolve();
    }

    return fetch("/predict_with_saved_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disk_model_name, text })
    })
    .then(res => res.json())
    .then(data => {
        const resultDiv = document.getElementById("test-predict-result");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color:red;">Lỗi dự đoán: ${data.error}</p>`;
        } else {
            let html = `<p><strong>Dự đoán:</strong> ${data.prediction}</p>`;
            if (data.probabilities && data.probabilities !== "N/A") {
                html += "<p><strong>Xác suất các lớp:</strong></p><ul>";
                for (const [cls, prob] of Object.entries(data.probabilities)) {
                    html += `<li>Lớp ${cls}: ${(prob * 100).toFixed(2)}%</li>`;
                }
                html += "</ul>";
            }
            resultDiv.innerHTML = html;
        }
    });
}

function generateComparisonsUI() {
    return fetch("/generate_model_comparisons", { method: "POST" })
    .then(res => res.json())
    .then(data => {
        const compareResultDiv = document.getElementById("compare-result");
        if (data.error) {
            compareResultDiv.innerHTML = `<p style="color:red;">Lỗi: ${data.error}</p>`;
            return;
        }
        let html = "";
        if (data.accuracy_chart_url) {
            html += `<h4>Biểu đồ so sánh Accuracy:</h4><img src="${data.accuracy_chart_url}" alt="Accuracy Comparison">`;
        }
        if (data.confusion_matrices && data.confusion_matrices.length > 0) {
            html += "<h4>Confusion Matrices của các Model Đã Lưu:</h4>";
            data.confusion_matrices.forEach(cm_info => {
                html += `<div style="display:inline-block; margin:5px; text-align:center;">
                            <p><em>${cm_info.name}</em></p>
                            <img src="${cm_info.url}" alt="CM for ${cm_info.name}" style="max-width:300px; border:1px solid #ccc;">
                         </div>`;
            });
        } else if (!data.accuracy_chart_url) {
            html = "<p><em>Không có dữ liệu để hiển thị so sánh.</em></p>";
        }
        compareResultDiv.innerHTML = html;
    });
}

function clearAllChartsAndSavedModels() {
    if (!confirm("Bạn có chắc chắn muốn xoá TẤT CẢ các model đã lưu và biểu đồ liên quan không? Hành động này không thể hoàn tác.")) {
        return Promise.resolve();
    }
    return fetch("/clear_all_saved_models_and_charts", { method: "POST" })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi xoá dữ liệu: " + data.error);
        } else {
            alert(data.message);
            loadSavedModelsForTesting(); // Refresh dropdown
            document.getElementById("compare-result").innerHTML = ""; // Clear comparison display
            document.getElementById("train-charts").innerHTML = ""; // Clear current train charts
            document.getElementById("train-result-text").innerText = ""; // Clear current train text
        }
    });
}

// ==== RECOMMENDATION TAB ====
function uploadRecommendationCSV() {
    const fileInput = document.getElementById("recommendCsvInput");
    if (!fileInput.files.length) {
        alert("Vui lòng chọn file CSV cho dataset gợi ý.");
        return Promise.resolve();
    }
    const formData = new FormData();
    formData.append("recommend_csv_file", fileInput.files[0]);

    return fetch("/upload_recommend_dataset", { method: "POST", body: formData })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi tải CSV gợi ý: " + data.error);
            return;
        }
        const selKeyword = document.getElementById("rec-keyword-col");
        const selProductId = document.getElementById("rec-product-id-col");
        selKeyword.innerHTML = data.columns.map(col => `<option value="${col}">${col}</option>`).join('');
        selProductId.innerHTML = data.columns.map(col => `<option value="${col}">${col}</option>`).join('');
        document.getElementById("recommend-columns-selection").style.display = "block";
        document.getElementById("recommend-train-section").style.display = "none";
        document.getElementById("recommend-query-section").style.display = "none";
    });
}

function selectRecommendColumns() {
    const keyword_col = document.getElementById("rec-keyword-col").value;
    const product_id_col = document.getElementById("rec-product-id-col").value;
    if (keyword_col === product_id_col) {
        alert("Cột mô tả và cột ID sản phẩm không được trùng nhau!");
        return;
    }
    return fetch("/confirm_recommend_cols", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ keyword_col, product_id_col })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi chọn cột gợi ý: " + data.error);
        } else {
            alert(data.message + " Sẵn sàng để Train SVD.");
            document.getElementById("recommend-train-section").style.display = "block";
            document.getElementById("recommend-query-section").style.display = "none";
            document.getElementById("svd-train-status").textContent = "";
        }
    });
}

function trainRecommendSVD() {
    document.getElementById("svd-train-status").textContent = "Đang huấn luyện SVD...";
    return fetch("/train_svd_recommender", { method: "POST" })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Lỗi huấn luyện SVD: " + data.error);
            document.getElementById("svd-train-status").textContent = `Lỗi: ${data.error}`;
            document.getElementById("recommend-query-section").style.display = "none";
        } else {
            alert(data.message);
            document.getElementById("svd-train-status").textContent = data.message;
            document.getElementById("recommend-query-section").style.display = "block";
        }
    });
}

function getProductRecommendations() {
    const query_text = document.getElementById("recommend-text-query").value.trim();
    if (!query_text) {
        alert("Vui lòng nhập mô tả để tìm sản phẩm.");
        return Promise.resolve();
    }
    return fetch("/get_svd_recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_text })
    })
    .then(res => res.json())
    .then(data => {
        const resultDiv = document.getElementById("recommend-result-display");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color:red;">Lỗi: ${data.error}</p>`;
        } else if (data.recommendations && data.recommendations.length > 0) {
            let html = "<h4>Sản phẩm gợi ý:</h4><ul>";
            data.recommendations.forEach(item => {
                html += `<li><strong>Sản phẩm:</strong> ${escapeHtml(item.product_id)} <br>
                             <em>Mô tả khớp:</em> ${escapeHtml(item.description_preview)} <br>
                             (Độ tương đồng: ${item.similarity_score.toFixed(3)})</li><br>`;
            });
            html += "</ul>";
            resultDiv.innerHTML = html;
        } else {
            resultDiv.innerHTML = "<p><em>Không tìm thấy sản phẩm nào phù hợp.</em></p>";
        }
    });
}

// ==== CHATBOT ====

function createTypingIndicatorHTML(botName, indicatorId) {
    return `
        <div class="typing-indicator-container" id="${indicatorId}">
            <div class="bot-message typing-indicator">
                <strong>${escapeHtml(botName)}:</strong>
                <div class="dot-flashing"></div>
            </div>
        </div>
    `;
}

function sendChatMessage() {
    const input = document.getElementById("chat-input");
    const chatLog = document.getElementById("chat-log");
    const userMessage = input.value.trim();

    if (!userMessage) return;

    // Hiển thị tin nhắn người dùng
    chatLog.innerHTML += `<div class="user-message"><strong>Bạn:</strong> ${escapeHtml(userMessage)}</div>`;
    input.value = "";

    // Tạo và thêm typing indicator
    const indicatorId = "default-bot-typing-active"; // ID tạm thời
    const indicatorHTML = createTypingIndicatorHTML("Bot", indicatorId);
    // Thêm indicator vào TRƯỚC khi cuộn và fetch
    const tempDiv = document.createElement('div'); // Tạo một div tạm để chứa HTML của indicator
    tempDiv.innerHTML = indicatorHTML;
    chatLog.appendChild(tempDiv.firstChild); // Thêm element indicator thực sự vào chatLog

    chatLog.scrollTop = chatLog.scrollHeight; // Cuộn xuống để thấy indicator

    fetch("/chat_message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage })
    })
    .then(res => res.json())
    .then(data => {
        chatLog.innerHTML += `<div class="bot-message"><strong>Bot:</strong> ${escapeHtml(data.reply || "...")}</div>`;
    })
    .catch(err => {
        chatLog.innerHTML += `<div class="bot-message"><strong>Bot:</strong> Lỗi: ${escapeHtml(err.message)}</div>`;
    })
    .finally(() => {
        // Xóa typing indicator
        const activeIndicator = document.getElementById(indicatorId);
        if (activeIndicator) {
            activeIndicator.remove();
        }
        chatLog.scrollTop = chatLog.scrollHeight; // Cuộn xuống sau khi có kết quả hoặc lỗi
    });
}

// Add event listener for Enter key in chat input
document.getElementById('chat-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
});

// ==== UTILITY RENDER HELPERS ====
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') {
        // If it's not a string, stringify it (e.g., for objects or numbers)
        // and then escape. This is a basic safeguard.
        unsafe = String(unsafe);
    }
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

function renderKeyValueTable(obj) {
    if (typeof obj !== 'object' || obj === null) return `<pre>${escapeHtml(JSON.stringify(obj, null, 2))}</pre>`;
    let html = '<table><tr><th>Key/Token</th><th>Value/Vector (mẫu)</th></tr>';
    for (const [key, value] of Object.entries(obj)) {
        html += `<tr><td>${escapeHtml(key)}</td><td>`;
        if (Array.isArray(value)) {
            html += `[${value.slice(0, 5).map(v => typeof v === 'number' ? v.toFixed(3) : escapeHtml(v)).join(', ')}${value.length > 5 ? ', ...' : ''}]`;
        } else {
            html += escapeHtml(String(value).substring(0,100));
        }
        html += `</td></tr>`;
    }
    html += '</table>';
    return html;
}

function renderVectorArray(vector, limit = 10) {
    if (!Array.isArray(vector)) return escapeHtml(String(vector));
    return `[${vector.slice(0, limit).map(v => v.toFixed ? v.toFixed(3) : escapeHtml(v)).join(', ')}${vector.length > limit ? ', ...' : ''}] (Chiều: ${vector.length})`;
}


// Initial setup when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    showTab('input', document.querySelector('.tab-btn')); // Show first tab and mark active
    updateModelParams(); // Initialize params for default model in Train tab
});

function sendGeminiChatMessage() {
    const input = document.getElementById("gemini-chat-input");
    const chatLog = document.getElementById("gemini-chat-log");
    const userMessage = input.value.trim();

    if (!userMessage) return;

    // Hiển thị tin nhắn người dùng
    chatLog.innerHTML += `<div class="user-message"><strong>Bạn:</strong> ${escapeHtml(userMessage)}</div>`;
    input.value = "";

    // Tạo và thêm typing indicator
    const indicatorId = "gemini-bot-typing-active"; // ID tạm thời
    const indicatorHTML = createTypingIndicatorHTML("Gemini Bot", indicatorId);
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = indicatorHTML;
    chatLog.appendChild(tempDiv.firstChild);

    chatLog.scrollTop = chatLog.scrollHeight; // Cuộn xuống để thấy indicator

    fetch("/gemini_chat_message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage })
    })
    .then(res => res.json())
    .then(data => {
        let botHtmlReply = "";
        if (data.reply) {
            if (marked && typeof marked.parse === 'function') {
                try {
                    botHtmlReply = marked.parse(data.reply, { breaks: true, gfm: true, async: false });
                } catch (e) {
                    console.error("Lỗi khi parse Markdown:", e);
                    botHtmlReply = escapeHtml(data.reply) + " (Lỗi parse Markdown)";
                }
            } else {
                botHtmlReply = escapeHtml(data.reply);
            }
        } else {
            botHtmlReply = escapeHtml("Xin lỗi, có lỗi xảy ra với Gemini.");
        }
        chatLog.innerHTML += `
        <div class="bot-message">
            <strong>Gemini Bot:</strong>
            ${botHtmlReply || ''}
        </div>`;
    })
    .catch(err => {
        chatLog.innerHTML += `<div class="bot-message"><strong>Gemini Bot:</strong> Lỗi Gemini: ${escapeHtml(err.message)}</div>`;
    })
    .finally(() => {
        // Xóa typing indicator
        const activeIndicator = document.getElementById(indicatorId);
        if (activeIndicator) {
            activeIndicator.remove();
        }
        chatLog.scrollTop = chatLog.scrollHeight; // Cuộn xuống sau khi có kết quả hoặc lỗi
    });
}

// Listener cho Enter key
document.getElementById('gemini-chat-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendGeminiChatMessage();
    }
});