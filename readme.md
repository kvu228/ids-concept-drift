# IDS Concept Drift - Nghiên cứu và Khắc phục Suy giảm Hiệu suất

Dự án nghiên cứu về **concept drift** trong hệ thống **Intrusion Detection System (IDS)** sử dụng dataset NSL-KDD. So sánh hiệu quả giữa mô hình tĩnh (Static) và mô hình adaptive (ARF - Adaptive Random Forest) trong việc khắc phục suy giảm hiệu suất do concept drift.

## 📋 Requirements

### Phiên bản cơ bản (`ids_concept_drift.py`)
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Phiên bản ARF với scikit-multiflow (`ids_concept_drift_ARF_new_v2.py`)
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
scikit-multiflow>=0.5.0
```

### Phiên bản ARF với River (Khuyến nghị) (`ids_concept_drift_ARF_new_v2_river_full.py`)
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
river>=0.21.0
```

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone repository hoặc tạo folder
```bash
mkdir ids_concept_drift_project
cd ids_concept_drift_project
```

### Bước 2: Tạo virtual environment (khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Download NSL-KDD Dataset

**Option 1: Download manual**
1. Truy cập: https://www.unb.ca/cic/datasets/nsl.html
2. Download 2 files:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`
3. Đặt vào folder project

**Option 2: Download bằng script**
```python
import urllib.request

# URLs
train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

# Download
urllib.request.urlretrieve(train_url, "KDDTrain+.txt")
urllib.request.urlretrieve(test_url, "KDDTest+.txt")
print("✅ Downloaded NSL-KDD dataset")
```

### Bước 5: Chạy code

**Lựa chọn phiên bản:**

1. **Phiên bản cơ bản** (Static vs Adaptive IDS):
```bash
python ids_concept_drift.py
```

2. **Phiên bản ARF với scikit-multiflow**:
```bash
pip install scikit-multiflow
python ids_concept_drift_ARF_new_v2.py
```

3. **Phiên bản ARF với River** (Khuyến nghị - dễ cài đặt hơn):
```bash
pip install river
python ids_concept_drift_ARF_new_v2_river_full.py
```

## 📁 Cấu trúc Project

```
ids-concept-drift/
│
├── ids_concept_drift.py                    # Phiên bản cơ bản (Static vs Adaptive)
├── ids_concept_drift_ARF_new_v2.py         # ARF với scikit-multiflow
├── ids_concept_drift_ARF_new_v2_river.py   # ARF với River (4 variants)
├── ids_concept_drift_ARF_new_v2_river_full.py  # ARF với River (7 variants) ⭐
├── requirements.txt                        # Dependencies cơ bản
├── pyproject.toml                          # Project config
├── README.md                               # Documentation
│
├── data/
│   ├── KDDTrain+.txt                      # Training data (~125K samples)
│   └── KDDTest+.txt                       # Test data (~22K samples)
│
├── results/
│   ├── result_arf/                        # Kết quả ARF experiments
│   ├── results_summary.csv                # Summary metrics (AA, FM, BWT)
│   └── *.png                              # Visualization plots
│
├── .cursor/
│   └── context/                           # Context files cho AI assistant
│       ├── 00-project-overview.md
│       ├── 01-architecture.md
│       ├── 02-source-files.md
│       ├── 03-data-flow.md
│       ├── 04-key-concepts.md
│       └── 05-implementation-details.md
│
└── .gitignore                             # Git ignore rules
```

## 📚 Các Phiên bản Code

### 1. `ids_concept_drift.py` - Phiên bản Cơ bản
**Mục đích**: Minh họa concept drift và adaptive learning cơ bản

**Tính năng**:
- Static IDS (RandomForest train 1 lần)
- Adaptive IDS (RandomForest với incremental update)
- Concept drift simulation (5 periods)
- So sánh Accuracy và F1-score

**Sử dụng khi**: Muốn hiểu cơ bản về concept drift và adaptive learning

---

### 2. `ids_concept_drift_ARF_new_v2.py` - ARF với scikit-multiflow
**Mục đích**: Implement ARF với đầy đủ drift detectors

**Tính năng**:
- 7 ARF variants (None, ADWIN, DDM, PageHinkley, KSWIN, HDDM_A, HDDM_W)
- Metrics: AA (Average Accuracy), FM (Forgetting Measure), BWT (Backward Transfer)
- Drift simulation bằng cách mix train/test data
- So sánh tất cả models

**Sử dụng khi**: Muốn nghiên cứu ARF với scikit-multiflow

**Lưu ý**: scikit-multiflow có thể khó cài đặt trên một số hệ thống

---

### 3. `ids_concept_drift_ARF_new_v2_river_full.py` - ARF với River ⭐
**Mục đích**: Phiên bản đầy đủ và dễ sử dụng nhất

**Tính năng**:
- 7 ARF variants với River library
- Metrics: AA, FM, BWT
- Visualization tất cả models
- Export results to CSV

**Sử dụng khi**: Muốn nghiên cứu ARF một cách đầy đủ và dễ dàng

**Ưu điểm**: River dễ cài đặt hơn scikit-multiflow, đặc biệt trên macOS

---

## 🔬 Metrics Đánh giá

### AA (Average Accuracy)
Độ chính xác trung bình trên tất cả các periods sau khi học xong period cuối cùng.

**Công thức**: `AA = mean(acc_matrix[-1, :])`

**Giá trị tốt**: Càng cao càng tốt (0-1)

### FM (Forgetting Measure)
Đo lường mức độ "quên" kiến thức cũ khi học task mới.

**Công thức**: `FM = mean_k(max_t(acc_matrix[:, k]) - acc_matrix[-1, k])`

**Giá trị tốt**: Càng thấp càng tốt (0-1)

### BWT (Backward Transfer)
Đo lường khả năng cải thiện performance trên các task trước đó nhờ học task mới.

**Công thức**: `BWT = mean_{k<T-1}(acc_matrix[-1, k] - acc_matrix[k, k])`

**Giá trị tốt**: Càng cao càng tốt (có thể âm)

## 🧪 Testing Code

### Test với NSL-KDD Dataset

**Bước 1**: Đảm bảo dataset đã được đặt trong folder `data/`:
- `data/KDDTrain+.txt`
- `data/KDDTest+.txt`

**Bước 2**: Chạy script tương ứng với phiên bản bạn muốn test:

```bash
# Phiên bản cơ bản
python ids_concept_drift.py

# Phiên bản ARF với River (khuyến nghị)
python ids_concept_drift_ARF_new_v2_river_full.py

# Phiên bản ARF với scikit-multiflow
python ids_concept_drift_ARF_new_v2.py
```

### Fallback với Sample Data
Nếu không có NSL-KDD dataset, phiên bản cơ bản (`ids_concept_drift.py`) sẽ tự động tạo sample data để demo.

## 📊 Expected Output

### Console Output
```
======================================================================
BÀI TẬP: CODING TÁI HIỆN & KHẮC PHỤC SUY GIẢM IDS
======================================================================

📥 Đang tải dữ liệu NSL-KDD...
✅ Train set: (125973, 43)
✅ Test set: (22544, 43)

🔄 Preprocessing data...
✅ Total data: (148517, 42)

🔄 Tạo 5 periods với concept drift...
  Period 1: 29703 samples, Attack rate: 53.46%
  Period 2: 29703 samples, Attack rate: 53.46%
  Period 3: 29703 samples, Attack rate: 53.46%
  Period 4: 29703 samples, Attack rate: 65.32%
  Period 5: 29705 samples, Attack rate: 65.32%

======================================================================
SO SÁNH STATIC IDS vs ADAPTIVE IDS
======================================================================

🔧 Training Static IDS...
✅ Static IDS trained

🔧 Initial training Adaptive IDS...
✅ Adaptive IDS initially trained

📊 PERIOD 1:
  Static IDS  - Accuracy: 0.9520, F1: 0.9445
  Adaptive IDS - Accuracy: 0.9520, F1: 0.9445

📊 PERIOD 2:
  Static IDS  - Accuracy: 0.9485, F1: 0.9401
  Adaptive IDS - Accuracy: 0.9512, F1: 0.9438
🔄 Updating model với 200 samples...

[...]

📉 PHÂN TÍCH SUY GIẢM HIỆU SUẤT
======================================================================

Static IDS:
  Accuracy ban đầu: 0.9520
  Accuracy cuối cùng: 0.8012
  📉 Suy giảm: 0.1508 (15.08%)

Adaptive IDS:
  Accuracy ban đầu: 0.9520
  Accuracy cuối cùng: 0.9156
  📉 Suy giảm: 0.0364 (3.64%)

✅ Adaptive IDS giảm suy giảm: 0.1144 (75.9%)

📊 Đã lưu biểu đồ: ids_concept_drift_comparison.png

======================================================================
✅ HOÀN THÀNH!
======================================================================
```

### Generated Files

**Phiên bản cơ bản**:
1. `ids_concept_drift_comparison.png` - Biểu đồ so sánh Static vs Adaptive
2. Console logs với metrics chi tiết

**Phiên bản ARF**:
1. `results_summary.csv` - Bảng tổng hợp metrics (AA, FM, BWT) cho tất cả models
2. Visualization plots - Accuracy và F1-score qua các periods
3. Console output với summary table được sắp xếp theo AA

## 🔧 Troubleshooting

### Lỗi 1: Module not found
```bash
# Cài đặt dependencies cơ bản
pip install numpy pandas scikit-learn matplotlib

# Cho phiên bản ARF với River
pip install river

# Cho phiên bản ARF với scikit-multiflow
pip install scikit-multiflow
```

### Lỗi 2: File not found (KDDTrain+.txt)
- Đảm bảo dataset được đặt trong folder `data/`
- Download dataset theo hướng dẫn Bước 4
- Hoặc sử dụng phiên bản cơ bản để tự động tạo sample data

### Lỗi 3: scikit-multiflow installation failed
**Giải pháp**: Sử dụng phiên bản River thay thế
```bash
pip install river
python ids_concept_drift_ARF_new_v2_river_full.py
```

### Lỗi 4: River import errors (DDM, HDDM_A, HDDM_W)
River có thể có các version khác nhau với import paths khác nhau. Code đã xử lý tự động với try-except blocks. Nếu vẫn lỗi:
```bash
pip install --upgrade river
```

### Lỗi 5: Memory error
- Giảm `period_size` trong `create_drift_periods()` (mặc định 8000)
- Giảm số periods (mặc định 5)
- Hoặc tăng RAM/swap

### Lỗi 6: Sklearn version incompatible
```bash
pip install --upgrade scikit-learn
```

## 📈 Customization

### Thay đổi số periods
```python
# Phiên bản cơ bản
data_with_drift = create_concept_drift(all_data, n_periods=10)  # Từ 5 → 10

# Phiên bản ARF
periods = create_drift_periods(
    X_train, y_train, X_test, y_test,
    n_periods=10,  # Từ 5 → 10
    period_size=8000
)
```

### Thay đổi drift schedule
```python
# Phiên bản ARF: Thay đổi tỷ lệ mix test data
schedule = [0.0, 0.1, 0.3, 0.6, 0.9]  # Custom schedule
periods = create_drift_periods(
    X_train, y_train, X_test, y_test,
    n_periods=5,
    period_size=8000,
    test_mix_schedule=schedule
)
```

### Thay đổi update frequency (phiên bản cơ bản)
```python
# Trong main()
adaptive_ids = AdaptiveIDS(update_frequency=500)  # Từ 200 → 500
```

### Thay đổi số models trong ARF
```python
# Trong build_arf_variants()
arf = forest.ARFClassifier(
    n_models=20,  # Từ 10 → 20 (nhiều trees hơn)
    drift_detector=ADWIN(),
    warning_detector=ADWIN(),
    seed=42
)
```

### Thêm ARF variants
```python
# Trong build_arf_variants()
def build_arf_variants():
    return {
        # ... existing variants ...
        "ARF_Custom": forest.ARFClassifier(
            n_models=15,
            drift_detector=CustomDetector(),
            seed=42
        ),
    }
```

### Thêm metrics khác
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Trong evaluate()
metrics['auc'] = roc_auc_score(y, y_pred_proba)
metrics['precision'], metrics['recall'], _ = precision_recall_curve(y, y_pred_proba)
```

## 🎯 Tips

1. **Code quality:**
   - Comments rõ ràng
   - Functions có docstrings
   - Code formatting chuẩn (PEP 8)

2. **Analysis depth:**
   - Giải thích tại sao results như vậy
   - So sánh với papers khác
   - Thảo luận limitations

3. **Visualization:**
   - Biểu đồ đẹp, rõ ràng
   - Có legends, labels đầy đủ
   - Multiple charts (accuracy, F1, confusion matrix)

4. **Report writing:**
   - Structure rõ ràng
   - Citations đầy đủ
   - Figures có captions
   - Tables formatted tốt

5. **GitHub repository:**
   - README.md chi tiết
   - Code organized tốt
   - .gitignore file
   - License file

## 🆘 Support

Nếu gặp vấn đề:
1. Check console error messages
2. Google error message
3. Check Stack Overflow

## 📚 Tài liệu tham khảo

### Concept Drift & Continual Learning
- **River ML**: https://riverml.xyz/latest/ - Thư viện online machine learning
- **scikit-multiflow**: https://scikit-multiflow.github.io/ - Multi-output streaming framework
- **ARF Paper**: Gomes et al. "Adaptive Random Forests for evolving data stream classification"

### NSL-KDD Dataset
- **Official Website**: https://www.unb.ca/cic/datasets/nsl.html
- **Original Paper**: Tavallaee et al. "A detailed analysis of the KDD CUP 99 data set" (2009)
- **GitHub Mirror**: https://github.com/defcom17/NSL_KDD

### Machine Learning Libraries
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Pandas**: https://pandas.pydata.org/
- **NumPy**: https://numpy.org/
- **Matplotlib**: https://matplotlib.org/

### Metrics & Evaluation
- **Continual Learning Metrics**: AA, FM, BWT được định nghĩa trong các papers về continual learning
- **Classification Metrics**: Accuracy, F1-score, Precision, Recall

## 📖 Context Files

Dự án bao gồm các file context trong `.cursor/context/` để hỗ trợ AI assistant hiểu rõ codebase:
- `00-project-overview.md` - Tổng quan dự án
- `01-architecture.md` - Kiến trúc và components
- `02-source-files.md` - Mô tả các file source code
- `03-data-flow.md` - Luồng dữ liệu và experiment pipeline
- `04-key-concepts.md` - Các khái niệm quan trọng
- `05-implementation-details.md` - Chi tiết implementation

---

**Good luck! 🚀**
