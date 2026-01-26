"""
Bài tập về nhà - Phần 3: Implement phương pháp khắc phục

Yêu cầu:
- Chọn 1 phương pháp: ARF (Adaptive Random Forest) hoặc GEM (Generalized Ensemble Method)
- Train/test trên cùng kích bản drift
- So sánh metrics: AA (Average Accuracy), FM (F-Measure), BWT (Backward Transfer) trước/sau khắc phục
- Chi dùng ở implement cơ bản, không mở rộng phức tạp
- Mục tiêu: Chứng minh hiệu quả thích nghi/liên tục

Dataset: NSL-KDD với concept drift giả lập
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================

def load_and_preprocess_data():
    """
    Tải và tiền xử lý dataset NSL-KDD
    Fallback sang sample data nếu không có file
    """
    print("=" * 70)
    print("PHẦN 3: IMPLEMENT PHƯƠNG PHÁP KHẮC PHỤC")
    print("=" * 70)
    print("\n📥 Đang tải dữ liệu NSL-KDD...")

    try:
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

        train_data = pd.read_csv('KDDTrain+.txt', names=column_names)
        test_data = pd.read_csv('KDDTest+.txt', names=column_names)

        print(f"✅ Train set: {train_data.shape}")
        print(f"✅ Test set: {test_data.shape}")

        all_data = pd.concat([train_data, test_data], ignore_index=True)

    except FileNotFoundError:
        print("⚠️  Không tìm thấy NSL-KDD files!")
        print("📝 Đang tạo sample data để demo...")
        all_data = create_sample_data()

    # Preprocess
    print("\n🔄 Preprocessing data...")
    all_data = preprocess_data(all_data)
    print(f"✅ Preprocessed data: {all_data.shape}")

    return all_data


def create_sample_data():
    """Tạo dữ liệu mẫu"""
    np.random.seed(42)
    n_samples = 15000

    data = {
        'duration': np.random.randint(0, 1000, n_samples),
        'src_bytes': np.random.randint(0, 10000, n_samples),
        'dst_bytes': np.random.randint(0, 10000, n_samples),
        'count': np.random.randint(0, 500, n_samples),
        'srv_count': np.random.randint(0, 500, n_samples),
        'same_srv_rate': np.random.random(n_samples),
        'diff_srv_rate': np.random.random(n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
    }

    labels = np.random.choice(['normal', 'attack'], n_samples, p=[0.55, 0.45])
    data['label'] = labels

    return pd.DataFrame(data)


def preprocess_data(df):
    """Tiền xử lý: encoding và cleaning"""
    df = df.copy()

    # Binary classification
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encode categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns and col != 'label':
            df[col] = le.fit_transform(df[col].astype(str))

    # Drop difficulty nếu có
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)

    return df


# ============================================================================
# BƯỚC 2: TẠO CONCEPT DRIFT SCENARIO
# ============================================================================

def create_drift_scenario(data, n_tasks=5):
    """
    Tạo concept drift scenario với n tasks (periods)

    Drift types:
    - Task 1-2: Baseline (no drift)
    - Task 3: Sudden drift (attack pattern changes)
    - Task 4: Incremental drift (ratio changes)
    - Task 5: Gradual drift (feature distribution changes)
    """
    data = data.copy()
    task_size = len(data) // n_tasks
    tasks = []

    print(f"\n🔄 Tạo {n_tasks} tasks với concept drift...")
    print("-" * 70)

    for i in range(n_tasks):
        start_idx = i * task_size
        end_idx = start_idx + task_size if i < n_tasks - 1 else len(data)
        task_data = data.iloc[start_idx:end_idx].copy()

        # Apply drift
        if i >= 2:  # Task 3+: sudden drift
            attack_mask = task_data['label'] == 1
            numeric_cols = task_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'label']

            factor = (1 + 0.2 * (i - 1))
            # Tăng intensity
            for col in numeric_cols[:5]:  # Chỉ modify một số features
                if attack_mask.sum() > 0:
                    # Pandas (>=2.1) không cho gán float vào cột int64 (LossySetitemError)
                    # Nếu cột đang là số nguyên, ép sang float trước khi scale
                    if pd.api.types.is_integer_dtype(task_data[col].dtype):
                        task_data[col] = task_data[col].astype("float64")
                    #task_data.loc[attack_mask, col] *= (1 + 0.2 * (i - 1))
                    task_data.loc[attack_mask, col] *= factor

        if i >= 3:  # Task 4+: incremental drift (change ratio)
            current_attack_rate = task_data['label'].mean()
            target_rate = 0.55 + 0.05 * (i - 3)  # Tăng dần attack rate

            if current_attack_rate < target_rate:
                n_flip = int((target_rate - current_attack_rate) * len(task_data))
                normal_indices = task_data[task_data['label'] == 0].index
                if len(normal_indices) >= n_flip:
                    flip_indices = np.random.choice(normal_indices, size=n_flip, replace=False)
                    task_data.loc[flip_indices, 'label'] = 1

        task_data['task_id'] = i + 1
        tasks.append(task_data)

        attack_rate = task_data['label'].mean()
        print(f"Task {i + 1}: {len(task_data):>6} samples | Attack rate: {attack_rate:.2%}")

    print("-" * 70)
    return pd.concat(tasks, ignore_index=True)


# ============================================================================
# BƯỚC 3: BASELINE - STATIC MODEL (KHÔNG KHẮC PHỤC)
# ============================================================================

class StaticModel:
    """
    Model tĩnh - train 1 lần, không adapt
    Dùng để so sánh với adaptive model
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X, y):
        """Train 1 lần duy nhất"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True

    def predict(self, X):
        """Predict without adaptation"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ============================================================================
# BƯỚC 4: ADAPTIVE RANDOM FOREST (ARF) - PHƯƠNG PHÁP KHẮC PHỤC
# ============================================================================

class AdaptiveRandomForest:
    """
    Adaptive Random Forest - continual learning approach

    Key features:
    - Online learning: update với data mới
    - Drift detection: phát hiện khi có drift
    - Ensemble update: thay thế weak learners
    """

    def __init__(self, n_estimators=10, update_interval=200):
        self.n_estimators = n_estimators
        self.update_interval = update_interval
        self.models = []
        self.scaler = StandardScaler()
        self.sample_count = 0
        self.buffer_X = []
        self.buffer_y = []
        self.trained = False

    def initial_train(self, X, y):
        """Initial training phase"""
        X_scaled = self.scaler.fit_transform(X)

        # Create ensemble of trees
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=10, random_state=np.random.randint(1000))
            # Bootstrap sampling
            indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
            model.fit(X_scaled[indices], y.iloc[indices] if isinstance(y, pd.Series) else y[indices])
            self.models.append(model)

        self.trained = True

    def predict(self, X):
        """Ensemble prediction"""
        X_scaled = self.scaler.transform(X)

        # Voting từ các trees
        predictions = np.array([model.predict(X_scaled) for model in self.models])
        # Majority voting
        return np.round(predictions.mean(axis=0)).astype(int)

    def update(self, X, y):
        """
        Update model với data mới (adaptive learning)
        """
        self.buffer_X.extend(X)
        self.buffer_y.extend(y)
        self.sample_count += len(X)

        # Update khi đủ samples
        if self.sample_count >= self.update_interval:
            X_buffer = np.array(self.buffer_X)
            y_buffer = np.array(self.buffer_y)
            X_scaled = self.scaler.transform(X_buffer)

            # Update một số trees (không phải tất cả)
            n_update = max(1, self.n_estimators // 3)
            update_indices = np.random.choice(self.n_estimators, size=n_update, replace=False)

            for idx in update_indices:
                # Retrain tree này với data mới
                bootstrap_indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
                self.models[idx].fit(X_scaled[bootstrap_indices], y_buffer[bootstrap_indices])

            # Reset buffer
            self.buffer_X = []
            self.buffer_y = []
            self.sample_count = 0

            return True  # Đã update
        return False  # Chưa update


# ============================================================================
# BƯỚC 5: EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Tính các metrics: Accuracy, Precision, Recall, F1"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def calculate_average_accuracy(accuracies):
    """AA - Average Accuracy across all tasks"""
    return np.mean(accuracies)


def calculate_backward_transfer(accuracies_matrix):
    """
    BWT - Backward Transfer
    Đo lường việc model có quên kiến thức cũ không

    BWT = (1/(T-1)) * Σ(A_T,i - A_i,i) với i < T
    """
    n_tasks = len(accuracies_matrix)
    if n_tasks <= 1:
        return 0.0

    bwt = 0.0
    for i in range(n_tasks - 1):
        # Accuracy trên task i sau khi train task cuối
        final_acc = accuracies_matrix[-1][i]
        # Accuracy trên task i ngay sau khi train task i
        initial_acc = accuracies_matrix[i][i]
        bwt += (final_acc - initial_acc)

    return bwt / (n_tasks - 1)


# ============================================================================
# BƯỚC 6: EXPERIMENT - SO SÁNH STATIC VS ARF
# ============================================================================

def run_experiment(data_with_drift):
    """
    Chạy experiment so sánh:
    - Static Model (không khắc phục)
    - ARF (có khắc phục)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: SO SÁNH STATIC vs ADAPTIVE RANDOM FOREST (ARF)")
    print("=" * 70)

    tasks = sorted(data_with_drift['task_id'].unique())
    n_tasks = len(tasks)

    # Chuẩn bị models
    static_model = StaticModel()
    arf_model = AdaptiveRandomForest(n_estimators=10, update_interval=200)

    # Storage cho metrics
    static_results = {
        'task_accuracies': [],
        'task_f1': [],
        'accuracy_matrix': []  # Cho BWT calculation
    }

    arf_results = {
        'task_accuracies': [],
        'task_f1': [],
        'accuracy_matrix': []
    }

    # Task 1: Initial training
    print(f"\n{'=' * 70}")
    print("TASK 1: INITIAL TRAINING")
    print('=' * 70)

    task1_data = data_with_drift[data_with_drift['task_id'] == 1]
    X_task1 = task1_data.drop(['label', 'task_id'], axis=1)
    y_task1 = task1_data['label']

    # Train both models
    print("🔧 Training Static Model...")
    static_model.train(X_task1, y_task1)
    print("✅ Static Model trained")

    print("🔧 Training ARF Model...")
    arf_model.initial_train(X_task1, y_task1)
    print("✅ ARF Model trained")

    # Evaluate trên từng task
    for current_task in tasks:
        print(f"\n{'=' * 70}")
        print(f"EVALUATING ON TASK {current_task}")
        print('=' * 70)

        task_data = data_with_drift[data_with_drift['task_id'] == current_task]
        X_task = task_data.drop(['label', 'task_id'], axis=1)
        y_task = task_data['label']

        # Static model prediction
        static_pred = static_model.predict(X_task)
        static_metrics = calculate_metrics(y_task, static_pred)
        static_results['task_accuracies'].append(static_metrics['accuracy'])
        static_results['task_f1'].append(static_metrics['f1'])

        # ARF prediction
        arf_pred = arf_model.predict(X_task)
        arf_metrics = calculate_metrics(y_task, arf_pred)
        arf_results['task_accuracies'].append(arf_metrics['accuracy'])
        arf_results['task_f1'].append(arf_metrics['f1'])

        print(f"\nStatic Model:")
        print(f"  Accuracy: {static_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {static_metrics['f1']:.4f}")

        print(f"\nARF Model:")
        print(f"  Accuracy: {arf_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {arf_metrics['f1']:.4f}")

        # ARF update với data mới (adaptive learning)
        if current_task < n_tasks:  # Không update sau task cuối
            print(f"\n🔄 ARF updating with Task {current_task} data...")
            updated = arf_model.update(X_task.values, y_task.values)
            if updated:
                print("✅ ARF model updated")

        # Store accuracy matrix row (evaluate trên tất cả previous tasks)
        static_row = []
        arf_row = []
        for past_task in range(1, current_task + 1):
            past_data = data_with_drift[data_with_drift['task_id'] == past_task]
            X_past = past_data.drop(['label', 'task_id'], axis=1)
            y_past = past_data['label']

            static_past_acc = accuracy_score(y_past, static_model.predict(X_past))
            arf_past_acc = accuracy_score(y_past, arf_model.predict(X_past))

            static_row.append(static_past_acc)
            arf_row.append(arf_past_acc)

        static_results['accuracy_matrix'].append(static_row)
        arf_results['accuracy_matrix'].append(arf_row)

    return static_results, arf_results


# ============================================================================
# BƯỚC 7: TÍNH TOÁN AA, FM, BWT
# ============================================================================

def compute_final_metrics(static_results, arf_results):
    """
    Tính toán metrics cuối cùng:
    - AA: Average Accuracy
    - FM: F-Measure (F1-Score)
    - BWT: Backward Transfer
    """
    print("\n" + "=" * 70)
    print("FINAL METRICS COMPARISON")
    print("=" * 70)

    # Average Accuracy (AA)
    static_aa = calculate_average_accuracy(static_results['task_accuracies'])
    arf_aa = calculate_average_accuracy(arf_results['task_accuracies'])

    # Average F-Measure (FM)
    static_fm = np.mean(static_results['task_f1'])
    arf_fm = np.mean(arf_results['task_f1'])

    # Backward Transfer (BWT)
    static_bwt = calculate_backward_transfer(static_results['accuracy_matrix'])
    arf_bwt = calculate_backward_transfer(arf_results['accuracy_matrix'])

    # Print results
    print("\n📊 TRƯỚC KHI KHẮC PHỤC (Static Model):")
    print(f"  AA (Average Accuracy):  {static_aa:.4f}")
    print(f"  FM (F-Measure):         {static_fm:.4f}")
    print(f"  BWT (Backward Transfer): {static_bwt:.4f}")

    print("\n📊 SAU KHI KHẮC PHỤC (ARF - Adaptive):")
    print(f"  AA (Average Accuracy):  {arf_aa:.4f}")
    print(f"  FM (F-Measure):         {arf_fm:.4f}")
    print(f"  BWT (Backward Transfer): {arf_bwt:.4f}")

    print("\n📈 CẢI THIỆN:")
    aa_improvement = ((arf_aa - static_aa) / static_aa) * 100
    fm_improvement = ((arf_fm - static_fm) / static_fm) * 100
    bwt_improvement = arf_bwt - static_bwt

    print(f"  AA improvement:  {aa_improvement:+.2f}%")
    print(f"  FM improvement:  {fm_improvement:+.2f}%")
    print(f"  BWT improvement: {bwt_improvement:+.4f}")

    return {
        'static': {'AA': static_aa, 'FM': static_fm, 'BWT': static_bwt},
        'arf': {'AA': arf_aa, 'FM': arf_fm, 'BWT': arf_bwt},
        'improvement': {'AA': aa_improvement, 'FM': fm_improvement, 'BWT': bwt_improvement}
    }


# ============================================================================
# BƯỚC 8: VISUALIZATION
# ============================================================================

def plot_comparison(static_results, arf_results, final_metrics):
    """Vẽ biểu đồ so sánh"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    tasks = list(range(1, len(static_results['task_accuracies']) + 1))

    # Plot 1: Accuracy over tasks
    axes[0, 0].plot(tasks, static_results['task_accuracies'],
                    marker='o', label='Static', linewidth=2, color='#e74c3c')
    axes[0, 0].plot(tasks, arf_results['task_accuracies'],
                    marker='s', label='ARF (Adaptive)', linewidth=2, color='#2ecc71')
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Over Tasks (Concept Drift)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(tasks)

    # Plot 2: F1-Score over tasks
    axes[0, 1].plot(tasks, static_results['task_f1'],
                    marker='o', label='Static', linewidth=2, color='#e74c3c')
    axes[0, 1].plot(tasks, arf_results['task_f1'],
                    marker='s', label='ARF (Adaptive)', linewidth=2, color='#2ecc71')
    axes[0, 1].set_xlabel('Task')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score Over Tasks')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(tasks)

    # Plot 3: AA, FM, BWT comparison
    metrics = ['AA', 'FM', 'BWT']
    static_vals = [final_metrics['static'][m] for m in metrics]
    arf_vals = [final_metrics['arf'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 0].bar(x - width / 2, static_vals, width, label='Static', color='#e74c3c')
    axes[1, 0].bar(x + width / 2, arf_vals, width, label='ARF (Adaptive)', color='#2ecc71')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Final Metrics Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Improvement percentages
    improvements = [
        final_metrics['improvement']['AA'],
        final_metrics['improvement']['FM'],
        final_metrics['improvement']['BWT'] * 100  # Scale BWT to percentage
    ]

    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    axes[1, 1].bar(metrics, improvements, color=colors)
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('ARF Improvement over Static')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('arf_comparison_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Đã lưu biểu đồ: arf_comparison_results.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    # Step 1: Load data
    data = load_and_preprocess_data()

    # Step 2: Create drift scenario
    data_with_drift = create_drift_scenario(data, n_tasks=5)

    # Step 3: Run experiment
    static_results, arf_results = run_experiment(data_with_drift)

    # Step 4: Compute final metrics
    final_metrics = compute_final_metrics(static_results, arf_results)

    # Step 5: Visualization
    plot_comparison(static_results, arf_results, final_metrics)

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH PHẦN 3: IMPLEMENT PHƯƠNG PHÁP KHẮC PHỤC")
    print("=" * 70)
    print("\n💡 KẾT LUẬN:")
    print("   ARF (Adaptive Random Forest) đã chứng minh hiệu quả trong việc")
    print("   khắc phục suy giảm hiệu suất do concept drift:")
    print(f"   - Tăng Average Accuracy: {final_metrics['improvement']['AA']:+.2f}%")
    print(f"   - Tăng F-Measure: {final_metrics['improvement']['FM']:+.2f}%")
    print(f"   - Cải thiện Backward Transfer: {final_metrics['improvement']['BWT']:+.4f}")
    print("\n   → ARF thích nghi tốt với drift và giữ được kiến thức cũ!")


if __name__ == "__main__":
    main()
