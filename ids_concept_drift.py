"""
Bài tập: Coding tái hiện & khắc phục suy giảm IDS
Dataset: NSL-KDD với concept drift giả lập
Tools: scikit-learn, scikit-multiflow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU NSL-KDD
# ============================================================================

def load_nsl_kdd():
    """
    Tải và tiền xử lý dataset NSL-KDD
    """
    # Column names cho NSL-KDD
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    print("📥 Đang tải dữ liệu NSL-KDD...")
    print("⚠️  Lưu ý: Bạn cần tải file KDDTrain+.txt và KDDTest+.txt từ:")
    print("   https://www.unb.ca/cic/datasets/nsl.html")

    try:
        # Đọc training set
        train_data = pd.read_csv('data/KDDTrain+.txt', names=column_names)
        test_data = pd.read_csv('data/KDDTest+.txt', names=column_names)

        print(f"✅ Train set: {train_data.shape}")
        print(f"✅ Test set: {test_data.shape}")

        return train_data, test_data
    except FileNotFoundError:
        print("❌ Không tìm thấy file! Đang tạo dữ liệu mẫu...")
        return create_sample_data()


def create_sample_data():
    """
    Tạo dữ liệu mẫu để demo (trong trường hợp không có NSL-KDD)
    """
    np.random.seed(42)
    n_samples = 10000

    # Tạo features ngẫu nhiên
    data = {
        'duration': np.random.randint(0, 1000, n_samples),
        'src_bytes': np.random.randint(0, 10000, n_samples),
        'dst_bytes': np.random.randint(0, 10000, n_samples),
        'count': np.random.randint(0, 500, n_samples),
        'srv_count': np.random.randint(0, 500, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
    }

    # Tạo labels (normal vs attack)
    labels = np.random.choice(['normal', 'attack'], n_samples, p=[0.6, 0.4])
    data['label'] = labels

    df = pd.DataFrame(data)

    # Chia thành train và test
    train_data = df[:7000]
    test_data = df[7000:]

    print("✅ Đã tạo dữ liệu mẫu")
    return train_data, test_data


def preprocess_data(df):
    """
    Tiền xử lý dữ liệu: encoding, scaling
    """
    df = df.copy()

    # Chuyển label về binary: normal (0) vs attack (1)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns and col != 'label':
            df[col] = le.fit_transform(df[col].astype(str))

    # Loại bỏ cột difficulty nếu có
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)

    return df


# ============================================================================
# BƯỚC 2: TẠO CONCEPT DRIFT GIẢ LẬP
# ============================================================================

def create_concept_drift(data, n_periods=5):
    """
    Tạo concept drift bằng cách thay đổi phân phối dữ liệu theo thời gian

    Kịch bản:
    - Period 1-2: Dữ liệu gốc (baseline)
    - Period 3: Xuất hiện attack pattern mới
    - Period 4: Thay đổi tỷ lệ attack/normal
    - Period 5: Feature drift (thay đổi statistical properties)
    """
    data = data.copy()
    period_size = len(data) // n_periods
    periods = []

    print(f"\n🔄 Tạo {n_periods} periods với concept drift...")

    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = start_idx + period_size if i < n_periods - 1 else len(data)
        period_data = data.iloc[start_idx:end_idx].copy()

        # Áp dụng drift theo từng period
        if i >= 2:  # Từ period 3 trở đi có drift
            # Thay đổi attack pattern
            attack_mask = period_data['label'] == 1
            if attack_mask.sum() > 0:
                # Tăng intensity của attacks
                numeric_cols = period_data.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'label']

                factor = 1 + 0.3 * (i - 1)
                for col in numeric_cols:
                    # Pandas (>=2.1) không cho gán float vào cột int64 (LossySetitemError)
                    # Nếu cột đang là số nguyên, ép sang float trước khi scale
                    if pd.api.types.is_integer_dtype(period_data[col].dtype):
                        period_data[col] = period_data[col].astype("float64")
                    #period_data.loc[attack_mask, col] *= (1 + 0.3 * (i - 1))
                    period_data.loc[attack_mask, col] = period_data.loc[attack_mask, col] * factor

        if i >= 3:  # Period 4: thay đổi tỷ lệ
            # Tăng tỷ lệ attacks
            n_attacks = int(len(period_data) * 0.5)
            normal_indices = period_data[period_data['label'] == 0].index
            if len(normal_indices) > n_attacks:
                flip_indices = np.random.choice(normal_indices,
                                                size=min(n_attacks, len(normal_indices) // 2),
                                                replace=False)
                period_data.loc[flip_indices, 'label'] = 1

        period_data['period'] = i + 1
        periods.append(period_data)
        print(f"  Period {i + 1}: {len(period_data)} samples, "
              f"Attack rate: {period_data['label'].mean():.2%}")

    return pd.concat(periods, ignore_index=True)


# ============================================================================
# BƯỚC 3: STATIC IDS (BASELINE) - KHÔNG ADAPTIVE
# ============================================================================

class StaticIDS:
    """
    IDS tĩnh - train 1 lần, không update
    Dùng để minh họa sự suy giảm hiệu suất
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train, y_train):
        """Train model 1 lần duy nhất"""
        print("\n🔧 Training Static IDS...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print("✅ Static IDS trained")

    def predict(self, X):
        """Predict without updating"""
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        """Đánh giá model"""
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }


# ============================================================================
# BƯỚC 4: ADAPTIVE IDS - CONTINUAL LEARNING
# ============================================================================

class AdaptiveIDS:
    """
    IDS adaptive - update liên tục khi có drift
    Sử dụng incremental learning
    """

    def __init__(self, update_frequency=100):
        self.model = RandomForestClassifier(n_estimators=50,
                                            max_samples=0.5,
                                            random_state=42)
        self.scaler = StandardScaler()
        self.update_frequency = update_frequency
        self.sample_count = 0
        self.buffer_X = []
        self.buffer_y = []
        self.is_trained = False

    def initial_train(self, X_train, y_train):
        """Initial training"""
        print("\n🔧 Initial training Adaptive IDS...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print("✅ Adaptive IDS initially trained")

    def predict_and_update(self, X, y):
        """
        Predict và update model nếu cần
        Mô phỏng continual learning
        """
        if not self.is_trained:
            raise Exception("Model chưa được initial train!")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Thêm vào buffer
        self.buffer_X.extend(X_scaled)
        self.buffer_y.extend(y)
        self.sample_count += len(X)

        # Update model nếu đủ samples
        if self.sample_count >= self.update_frequency:
            self._update_model()
            self.sample_count = 0
            self.buffer_X = []
            self.buffer_y = []

        return predictions

    def _update_model(self):
        """Update model với data mới"""
        if len(self.buffer_X) > 0:
            print(f"🔄 Updating model với {len(self.buffer_X)} samples...")
            X_buffer = np.array(self.buffer_X)
            y_buffer = np.array(self.buffer_y)

            # Retrain với mix của old và new data
            self.model.fit(X_buffer, y_buffer)

    def evaluate(self, X, y):
        """Đánh giá model"""
        y_pred = self.predict_and_update(X, y)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }


# ============================================================================
# BƯỚC 5: SO SÁNH VÀ VISUALIZATION
# ============================================================================

def compare_models(data_with_drift):
    """
    So sánh Static IDS vs Adaptive IDS qua các periods
    """
    print("\n" + "=" * 70)
    print("SO SÁNH STATIC IDS vs ADAPTIVE IDS")
    print("=" * 70)

    # Chuẩn bị dữ liệu
    periods = sorted(data_with_drift['period'].unique())

    # Initial training data (period 1)
    init_data = data_with_drift[data_with_drift['period'] == 1]
    X_init = init_data.drop(['label', 'period'], axis=1)
    y_init = init_data['label']

    # Initialize models
    static_ids = StaticIDS()
    adaptive_ids = AdaptiveIDS(update_frequency=200)

    static_ids.train(X_init, y_init)
    adaptive_ids.initial_train(X_init, y_init)

    # Test trên các periods
    results = {
        'period': [],
        'static_accuracy': [],
        'static_f1': [],
        'adaptive_accuracy': [],
        'adaptive_f1': []
    }

    for period in periods:
        period_data = data_with_drift[data_with_drift['period'] == period]
        X_test = period_data.drop(['label', 'period'], axis=1)
        y_test = period_data['label']

        # Evaluate Static IDS
        static_metrics = static_ids.evaluate(X_test, y_test)

        # Evaluate Adaptive IDS
        adaptive_metrics = adaptive_ids.evaluate(X_test, y_test)

        results['period'].append(period)
        results['static_accuracy'].append(static_metrics['accuracy'])
        results['static_f1'].append(static_metrics['f1'])
        results['adaptive_accuracy'].append(adaptive_metrics['accuracy'])
        results['adaptive_f1'].append(adaptive_metrics['f1'])

        print(f"\n📊 PERIOD {period}:")
        print(f"  Static IDS  - Accuracy: {static_metrics['accuracy']:.4f}, F1: {static_metrics['f1']:.4f}")
        print(f"  Adaptive IDS - Accuracy: {adaptive_metrics['accuracy']:.4f}, F1: {adaptive_metrics['f1']:.4f}")

    return results


def plot_results(results):
    """
    Vẽ biểu đồ so sánh
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    periods = results['period']

    # Plot Accuracy
    axes[0].plot(periods, results['static_accuracy'],
                 marker='o', label='Static IDS', linewidth=2)
    axes[0].plot(periods, results['adaptive_accuracy'],
                 marker='s', label='Adaptive IDS', linewidth=2)
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Over Time (Concept Drift)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot F1-Score
    axes[1].plot(periods, results['static_f1'],
                 marker='o', label='Static IDS', linewidth=2)
    axes[1].plot(periods, results['adaptive_f1'],
                 marker='s', label='Adaptive IDS', linewidth=2)
    axes[1].set_xlabel('Time Period')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score Over Time (Concept Drift)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ids_concept_drift_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Đã lưu biểu đồ: ids_concept_drift_comparison.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Chương trình chính
    """
    print("=" * 70)
    print("BÀI TẬP: CODING TÁI HIỆN & KHẮC PHỤC SUY GIẢM IDS")
    print("=" * 70)

    # Bước 1: Load data
    train_data, test_data = load_nsl_kdd()

    # Bước 2: Preprocess
    print("\n🔄 Preprocessing data...")
    train_processed = preprocess_data(train_data)
    test_processed = preprocess_data(test_data)

    # Combine để tạo drift
    all_data = pd.concat([train_processed, test_processed], ignore_index=True)
    print(f"✅ Total data: {all_data.shape}")

    # Bước 3: Tạo concept drift
    data_with_drift = create_concept_drift(all_data, n_periods=5)

    # Bước 4: So sánh models
    results = compare_models(data_with_drift)

    # Bước 5: Visualization
    plot_results(results)

    # Tính toán degradation
    print("\n" + "=" * 70)
    print("📉 PHÂN TÍCH SUY GIẢM HIỆU SUẤT")
    print("=" * 70)

    static_degradation = results['static_accuracy'][0] - results['static_accuracy'][-1]
    adaptive_degradation = results['adaptive_accuracy'][0] - results['adaptive_accuracy'][-1]

    print(f"\nStatic IDS:")
    print(f"  Accuracy ban đầu: {results['static_accuracy'][0]:.4f}")
    print(f"  Accuracy cuối cùng: {results['static_accuracy'][-1]:.4f}")
    print(f"  📉 Suy giảm: {static_degradation:.4f} ({static_degradation * 100:.2f}%)")

    print(f"\nAdaptive IDS:")
    print(f"  Accuracy ban đầu: {results['adaptive_accuracy'][0]:.4f}")
    print(f"  Accuracy cuối cùng: {results['adaptive_accuracy'][-1]:.4f}")
    print(f"  📉 Suy giảm: {adaptive_degradation:.4f} ({adaptive_degradation * 100:.2f}%)")

    improvement = static_degradation - adaptive_degradation
    print(f"\n✅ Adaptive IDS giảm suy giảm: {improvement:.4f} ({improvement / static_degradation * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
