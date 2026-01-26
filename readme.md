# HƯỚNG DẪN SETUP DỰ ÁN

## 📋 Requirements

### requirements.txt
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
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
```bash
python ids_concept_drift.py
```

## 📁 Cấu trúc Project

```
ids_concept_drift_project/
│
├── ids_concept_drift.py          # Main implementation
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
│
├── data/
│   ├── KDDTrain+.txt             # Training data
│   └── KDDTest+.txt              # Test data
│
├── results/
│   ├── ids_concept_drift_comparison.png
│   ├── confusion_matrices.png
│   └── performance_metrics.csv
│
└── report/
    └── report.pdf                 # Báo cáo cuối cùng
```

## 🧪 Testing Code

### Test nhanh (với sample data)
Nếu chưa có NSL-KDD, code sẽ tự động tạo sample data để demo.

```bash
python ids_concept_drift.py
```

### Test với NSL-KDD đầy đủ
Đặt 2 files `KDDTrain+.txt` và `KDDTest+.txt` vào folder project.

```bash
python ids_concept_drift.py
```

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
1. `ids_concept_drift_comparison.png` - Biểu đồ so sánh
2. Console logs với metrics chi tiết

## 🔧 Troubleshooting

### Lỗi 1: Module not found
```bash
pip install <missing_module>
```

### Lỗi 2: File not found (KDDTrain+.txt)
- Download dataset theo hướng dẫn Bước 4
- Hoặc để code tự tạo sample data

### Lỗi 3: Memory error
- Giảm kích thước dataset
- Hoặc tăng RAM/swap

### Lỗi 4: Sklearn version incompatible
```bash
pip install --upgrade scikit-learn
```

## 📈 Customization

### Thay đổi số periods
```python
# Trong main()
data_with_drift = create_concept_drift(all_data, n_periods=10)  # Từ 5 → 10
```

### Thay đổi update frequency
```python
# Trong main()
adaptive_ids = AdaptiveIDS(update_frequency=500)  # Từ 200 → 500
```

### Thay đổi model
```python
# Trong class StaticIDS hoặc AdaptiveIDS
from sklearn.svm import SVC
self.model = SVC(kernel='rbf')  # Thay vì RandomForest
```

### Thêm metrics khác
```python
from sklearn.metrics import roc_auc_score

# Trong evaluate()
metrics['auc'] = roc_auc_score(y, y_pred_proba)
```

## 📝 Checklist hoàn thành bài tập

- [ ] Code chạy thành công
- [ ] Có biểu đồ visualization
- [ ] Console output đầy đủ metrics
- [ ] Code có comments đầy đủ
- [ ] Báo cáo 10-15 trang
- [ ] Upload code lên GitHub
- [ ] README.md đầy đủ
- [ ] requirements.txt

## 🎯 Tips để có điểm cao

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
4. Hỏi bạn cùng lớp
5. Hỏi giảng viên

## 📚 Tài liệu tham khảo thêm

**Concept Drift:**
- https://riverml.xyz/latest/
- https://scikit-multiflow.github.io/

**NSL-KDD:**
- https://www.unb.ca/cic/datasets/nsl.html
- Original paper: Tavallaee et al. 2009

**Scikit-learn:**
- https://scikit-learn.org/stable/
- User guide: https://scikit-learn.org/stable/user_guide.html

---

**Good luck! 🚀**
