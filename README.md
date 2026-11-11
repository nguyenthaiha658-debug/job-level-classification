# Phân loại cấp bậc tin tuyển dụng

**Job-level classification from posting text**

## English summary

This project classifies the career level of an English-language job posting from its title, description, location, function and industry. It starts from an exercise in the Viet Nguyen AI machine-learning course and is rewritten as a small, reproducible experiment: a scikit-learn / imbalanced-learn pipeline, a fair comparison of Random Forest, Logistic Regression and LinearSVC, and three ways of dealing with a heavily imbalanced six-class target (`class_weight='balanced'`, `RandomOverSampler`, and `SMOTE`, all inside an `imblearn` pipeline so resampling happens only on training folds).

On a held-out test set of 1,615 postings, a course-style Random Forest baseline reaches **75.85% accuracy** but only **0.375 macro F1**, because it almost never predicts the rare classes. Logistic Regression with SMOTE reaches **0.654 macro F1** and **73.50% accuracy**. The accuracy drop is small; the gain comes from much better recall on mid- and rare classes. The smallest class has only four examples in the whole dataset (one in the test set), so its perfect F1 in that run is not a reliable result and is called out below.

This work is based on exercises from the Viet Nguyen AI ML course.

## Bài toán

Cho một tin tuyển dụng tiếng Anh với các trường `title`, `description`, `location`, `function` và `industry`, hãy dự đoán `career_level` (6 lớp). Đây là bài phân loại văn bản cổ điển, khó chủ yếu vì **mất cân bằng lớp** và vì các cấp bậc kề nhau (chuyên gia cao cấp / quản lý / trưởng bộ phận) dùng ngôn ngữ khá giống nhau.

## Dữ liệu

Bảng `final_project.ods` đi kèm khóa học Viet Nguyen AI: 8,074 dòng, sáu cột. Một dòng bị loại vì thiếu mô tả, còn **8,073** mẫu.

| `career_level` | Số mẫu | Tỷ lệ |
|---|---:|---:|
| senior_specialist_or_project_manager | 4,337 | 53.7% |
| manager_team_leader | 2,672 | 33.1% |
| bereichsleiter | 960 | 11.9% |
| director_business_unit_leader | 70 | 0.87% |
| specialist | 30 | 0.37% |
| managing_director_small_medium_company | 4 | 0.05% |

File gốc khoảng 10 MB nên **không được commit**. Cách lấy dữ liệu nằm trong [`data/README.md`](data/README.md).

Chia train/test 80/20, `stratify` theo nhãn, `random_state=42`: 6,458 / 1,615 mẫu. Lớp nhỏ nhất có 3 mẫu ở tập train và 1 mẫu ở tập test.

## Phương pháp

- Làm sạch `location`: với các chuỗi dạng `Houston, TX` chỉ giữ mã bang (`TX`). Regex được viết bằng raw string để không còn cảnh báo escape như trong script gốc.
- Đặc trưng (cùng họ đặc trưng với bài tập, viết lại cho gọn):
  - TF-IDF unigram trên `title` và `industry`
  - TF-IDF unigram + bigram trên `description`, `min_df=0.01`, `max_df=0.95`
  - One-hot trên `location` và `function` (`handle_unknown='ignore'`)
- Chọn đặc trưng bằng `SelectPercentile(chi2)`, lưới nhỏ `{5, 10}`.
- Mô hình: Random Forest (baseline giống hướng của bài tập), Logistic Regression, LinearSVC.
- Xử lý mất cân bằng (đây là phần mở rộng chính):
  1. không làm gì (baseline)
  2. `class_weight='balanced'`
  3. `RandomOverSampler` hoặc `SMOTE(k_neighbors=1)` đặt **trong** `imblearn.pipeline.Pipeline`, sau bước biến đổi, để chỉ resample các fold huấn luyện. `k_neighbors=1` vì lớp nhỏ nhất không đủ mẫu cho giá trị mặc định là 5.
- `GridSearchCV` 3-fold, stratified, tối ưu **macro F1** (phù hợp hơn accuracy hay `recall_weighted` khi các lớp hiếm mới là chỗ mô hình yếu). Lưới của mô hình tuyến tính thêm `C ∈ {0.5, 2.0}`.

## Kết quả

Số liệu dưới đây là kết quả chạy thật trên tập test (không phải số viết tay). Macro F1 là tiêu chí chọn mô hình tốt nhất.

| Thí nghiệm | Mô hình | Xử lý mất cân bằng | Accuracy | Macro F1 | Weighted F1 | Tham số tốt nhất (CV) |
|---|---|---|---:|---:|---:|---|
| `rf_baseline` | Random Forest | không | 0.7585 | 0.3746 | 0.7322 | percentile=5 |
| `rf_balanced` | Random Forest | `class_weight='balanced'` | 0.7511 | 0.4425 | 0.7510 | percentile=5 |
| `rf_ros` | Random Forest | RandomOverSampler | 0.7548 | 0.4402 | 0.7415 | percentile=5 |
| `logreg_balanced` | Logistic Regression | `class_weight='balanced'` | 0.7245 | 0.6280 | 0.7328 | C=2.0, percentile=10 |
| **`logreg_smote`** | **Logistic Regression** | **SMOTE (k=1)** | **0.7350** | **0.6535** | **0.7420** | **C=2.0, percentile=10** |
| `linearsvc_balanced` | LinearSVC | `class_weight='balanced'` | 0.7492 | 0.6364 | 0.7489 | C=2.0, percentile=10 |
| `linearsvc_ros` | LinearSVC | RandomOverSampler | 0.7276 | 0.6272 | 0.7334 | C=0.5, percentile=10 |

![Ma trận nhầm lẫn của mô hình tốt nhất](results/confusion_matrix.png)

Quan sát chính:

- Baseline Random Forest có **accuracy cao nhất** (75.85%) nhưng macro F1 chỉ 0.375: mô hình gần như không dự đoán `specialist` và `managing_director_...` (F1 = 0), và recall của `bereichsleiter` chỉ 0.146.
- Cân bằng trọng số hoặc oversampling giúp Random Forest tăng macro F1 lên khoảng 0.44, vẫn bỏ sót hai lớp hiếm nhất.
- Các mô hình tuyến tính hơn hẳn trên macro F1 (0.63–0.65). `logreg_smote` tốt nhất theo macro F1; `linearsvc_balanced` gần kề (0.636) và giữ accuracy cao hơn (74.92%).
- Phần lớn nhầm lẫn nằm giữa các cấp kề nhau (senior specialist ↔ manager ↔ bereichsleiter), đúng với việc mô tả công việc của các cấp này chồng lên nhau.
- **Lưu ý về lớp nhỏ nhất:** cả bộ dữ liệu chỉ có 4 mẫu `managing_director_small_medium_company`, tập test có đúng 1 mẫu. F1 = 1.00 của lớp này ở `logreg_smote` không có ý nghĩa thống kê; chỉ cần lần chạy khác là số macro F1 sẽ đổi. Phần cải thiện đáng tin hơn là recall của `bereichsleiter` (0.146 → 0.625) và `specialist` (0 → 0.667, dù chỉ có 6 mẫu test). Macro F1 trên 5 lớp còn lại (bỏ MD) vẫn cao hơn baseline rõ rệt.

Báo cáo từng lớp nằm trong `results/classification_reports/`.

## Những gì khác với script gốc của khóa học

- Viết lại thành repo có cấu trúc (`src/`, script thí nghiệm, `requirements.txt`), không phải sửa nhẹ file `job_cls.py`.
- Sửa regex location (raw string) để hết cảnh báo escape.
- So sánh thêm Logistic Regression và LinearSVC, không chỉ Random Forest.
- Đưa oversampling vào đúng chỗ (pipeline, chỉ trên fold huấn luyện), thay vì resample cả tập train một lần rồi mới cross-validate.
- Đổi metric tối ưu sang macro F1 và báo cáo đủ accuracy / macro F1 / weighted F1.
- Giữ lưới siêu tham số nhỏ để chạy được trên máy cá nhân (toàn bộ 7 thí nghiệm xong trong khoảng 2 phút trên máy này).

## Cách chạy

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# đặt final_project.ods vào data/ hoặc export JOB_DATA_PATH=...
python scripts/run_experiments.py
```

Kết quả được ghi vào `results/metrics.csv`, `results/confusion_matrix.png` và `results/classification_reports/`.

## Nguồn

Dựa trên bài tập của khóa học machine learning của Viet Nguyen AI. Phần thí nghiệm, cấu trúc code, sửa lỗi và phân tích kết quả được làm thêm trên nền bài tập đó.
