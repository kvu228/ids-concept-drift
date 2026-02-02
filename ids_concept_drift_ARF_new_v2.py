"""
ids_concept_drift_ARF_new_v2.py

Mục tiêu (bám đúng yêu cầu slide):
- Chọn 1 phương pháp khắc phục: ARF (Adaptive Random Forest)
- Train/test trên cùng kịch bản drift (tạo periods bằng cách mix KDDTrain+ và KDDTest+ theo tỉ lệ)
- So sánh metrics: AA, FM, BWT trước/sau (Static vs ARF variants)
- Implement cơ bản + rõ ràng, dễ giải thích

Chạy:
  python ids_concept_drift_ARF_new_v2.py

Yêu cầu dữ liệu:
  data/KDDTrain+.txt
  data/KDDTest+.txt

Yêu cầu thư viện:
  pip install numpy pandas scikit-learn matplotlib scikit-multiflow
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# --- scikit-multiflow ARF + detectors (như notebook) ---
try:
    from skmultiflow.meta import AdaptiveRandomForestClassifier
    from skmultiflow.drift_detection import PageHinkley, ADWIN, KSWIN, HDDM_A, HDDM_W, DDM
except Exception as e:
    raise RuntimeError(
        "Thiếu scikit-multiflow. Cài bằng: pip install scikit-multiflow\n"
        f"Chi tiết lỗi: {e}"
    )

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# 1) Load + preprocess NSL-KDD
# -----------------------------
def load_nsl_kdd(train_path: str, test_path: str):
    # NSL-KDD format: 41 features + label + difficulty = 43 columns
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Không thấy file train: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Không thấy file test: {test_path}")

    df_train = pd.read_csv(train_path, names=col_names, header=None)
    df_test = pd.read_csv(test_path, names=col_names, header=None)

    return df_train, df_test


def preprocess_binary(df: pd.DataFrame):
    """
    - label: 'normal' => 0, còn lại => 1
    - encode 3 cột categorical: protocol_type, service, flag
    - ép numeric + fillna
    """
    df = df.copy()

    # binary label
    df["label"] = df["label"].astype(str).str.lower()
    df["label"] = df["label"].apply(lambda x: 0 if "normal" in x else 1).astype(int)

    # encode categoricals
    cat_cols = ["protocol_type", "service", "flag"]
    for c in cat_cols:
        df[c] = df[c].astype(str)
        df[c] = pd.factorize(df[c])[0]

    # drop difficulty (không dùng)
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    # coerce all to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.fillna(0.0)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


# -------------------------------------------
# 2) Drift scenario: mix train/test by period
# -------------------------------------------
def create_drift_periods(
    X_train, y_train, X_test, y_test,
    n_periods=5,
    period_size=8000,
    test_mix_schedule=None
):
    """
    Tạo kịch bản drift bằng cách tăng dần tỷ lệ lấy mẫu từ test (được xem như phân phối mới).
    - period 1: mostly train
    - period cuối: mostly test

    test_mix_schedule: list tỷ lệ test cho mỗi period. Nếu None: tự tạo tuyến tính.
      Ví dụ: [0.0, 0.25, 0.5, 0.75, 0.9]
    """
    if test_mix_schedule is None:
        test_mix_schedule = np.linspace(0.0, 0.9, n_periods).tolist()

    assert len(test_mix_schedule) == n_periods, "test_mix_schedule phải có đúng n_periods phần tử"

    periods = []
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    for i in range(n_periods):
        p_test = float(test_mix_schedule[i])
        n_from_test = int(period_size * p_test)
        n_from_train = period_size - n_from_test

        idx_tr = np.random.choice(n_train, size=n_from_train, replace=(n_from_train > n_train))
        idx_te = np.random.choice(n_test, size=n_from_test, replace=(n_from_test > n_test))

        Xp = np.vstack([X_train[idx_tr], X_test[idx_te]])
        yp = np.concatenate([y_train[idx_tr], y_test[idx_te]])

        # shuffle within period
        perm = np.random.permutation(period_size)
        Xp = Xp[perm]
        yp = yp[perm]

        periods.append((Xp, yp))

    return periods


# ------------------------------------------
# 3) Build ARF variants (from your notebook)
# ------------------------------------------
def build_arf_variants():
    """
    Gợi ý: report chính chỉ cần 2-3 biến thể (ARF_None, ARF_ADWIN, ...)
    Nhưng để bạn "gọi đủ phương pháp" như notebook thì mình tạo hết ở đây.
    """
    return {
        "ARF_None": AdaptiveRandomForestClassifier(
            drift_detection_method=None,
            warning_detection_method=None,
            random_state=RANDOM_SEED
        ),
        "ARF_PageHinkley": AdaptiveRandomForestClassifier(
            drift_detection_method=PageHinkley(),
            warning_detection_method=PageHinkley(),
            random_state=RANDOM_SEED
        ),
        "ARF_ADWIN": AdaptiveRandomForestClassifier(
            drift_detection_method=ADWIN(),
            warning_detection_method=ADWIN(),
            random_state=RANDOM_SEED
        ),
        "ARF_KSWIN": AdaptiveRandomForestClassifier(
            drift_detection_method=KSWIN(),
            warning_detection_method=KSWIN(),
            random_state=RANDOM_SEED
        ),
        "ARF_HDDM_A": AdaptiveRandomForestClassifier(
            drift_detection_method=HDDM_A(),
            warning_detection_method=HDDM_A(),
            random_state=RANDOM_SEED
        ),
        "ARF_HDDM_W": AdaptiveRandomForestClassifier(
            drift_detection_method=HDDM_W(),
            warning_detection_method=HDDM_W(),
            random_state=RANDOM_SEED
        ),
        "ARF_DDM": AdaptiveRandomForestClassifier(
            drift_detection_method=DDM(),
            warning_detection_method=DDM(),
            random_state=RANDOM_SEED
        ),
    }


# ------------------------------------------
# 4) Continual metrics: AA / FM / BWT
# ------------------------------------------
def compute_AA_FM_BWT(acc_matrix):
    """
    acc_matrix[t][k] = accuracy on task k after learning up to task t
    T x T matrix

    AA: mean of last row
    FM: average forgetting = mean_k( max_t A[t,k] - A[T-1,k] )
    BWT: mean_{k<T-1}( A[T-1,k] - A[k,k] )
    """
    A = np.array(acc_matrix, dtype=float)
    T = A.shape[0]
    if T == 0:
        return 0.0, 0.0, 0.0

    AA = float(np.mean(A[-1, :]))

    FM_list = []
    for k in range(T):
        FM_list.append(float(np.max(A[:, k]) - A[-1, k]))
    FM = float(np.mean(FM_list))

    if T > 1:
        BWT = float(np.mean([A[-1, k] - A[k, k] for k in range(T - 1)]))
    else:
        BWT = 0.0

    return AA, FM, BWT


# ------------------------------------------
# 5) Experiment runner
# ------------------------------------------
def eval_acc_f1(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred), f1_score(y, y_pred)


def run_static(periods):
    """
    Static baseline:
    - Train once on period 1
    - Evaluate on each period (no update)
    """
    X1, y1 = periods[0]
    static = DecisionTreeClassifier(random_state=RANDOM_SEED)
    static.fit(X1, y1)

    acc_over_time, f1_over_time = [], []
    for (Xt, yt) in periods:
        acc, f1 = eval_acc_f1(static, Xt, yt)
        acc_over_time.append(acc)
        f1_over_time.append(f1)

    # Optional: build a "pseudo acc_matrix" (static doesn't learn over time)
    # We'll set acc_matrix rows identical = performance fixed after training on task1.
    T = len(periods)
    acc_matrix = []
    for t in range(T):
        row = []
        for k in range(T):
            row.append(acc_over_time[k])  # same row
        acc_matrix.append(row)

    AA, FM, BWT = compute_AA_FM_BWT(acc_matrix)

    return {
        "acc_over_time": acc_over_time,
        "f1_over_time": f1_over_time,
        "acc_matrix": acc_matrix,
        "AA": AA,
        "FM": FM,
        "BWT": BWT
    }


def run_arf_variants(periods, variants: dict):
    """
    For each ARF:
    - Incremental learn per task with partial_fit
    - After learning task t, evaluate on all tasks k to fill acc_matrix[t][k]
    """
    results = {}
    T = len(periods)
    classes = np.array([0, 1])

    for name, arf in variants.items():
        acc_matrix = [[0.0] * T for _ in range(T)]
        acc_over_time = []
        f1_over_time = []

        for t in range(T):
            Xt, yt = periods[t]

            # update
            if t == 0:
                arf.partial_fit(Xt, yt, classes=classes)
            else:
                arf.partial_fit(Xt, yt)

            # evaluate on all tasks for matrix
            for k in range(T):
                Xk, yk = periods[k]
                y_pred = arf.predict(Xk)
                acc_matrix[t][k] = accuracy_score(yk, y_pred)

            # track current-task metrics
            acc_over_time.append(acc_matrix[t][t])
            y_pred_t = arf.predict(Xt)
            f1_over_time.append(f1_score(yt, y_pred_t))

        AA, FM, BWT = compute_AA_FM_BWT(acc_matrix)
        results[name] = {
            "acc_over_time": acc_over_time,
            "f1_over_time": f1_over_time,
            "acc_matrix": acc_matrix,
            "AA": AA,
            "FM": FM,
            "BWT": BWT
        }

    return results


# ------------------------------------------
# 6) Plotting helpers
# ------------------------------------------
def plot_compare_two(periods, static_res, arf_res, title_prefix=""):
    T = len(periods)
    x = np.arange(1, T + 1)

    # Accuracy over time
    plt.figure(figsize=(10, 4))
    plt.plot(x, static_res["acc_over_time"], marker="o", label="Static")
    plt.plot(x, arf_res["acc_over_time"], marker="s", label="ARF")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Period")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}Accuracy Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # F1 over time
    plt.figure(figsize=(10, 4))
    plt.plot(x, static_res["f1_over_time"], marker="o", label="Static")
    plt.plot(x, arf_res["f1_over_time"], marker="s", label="ARF")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Period")
    plt.ylabel("F1-score")
    plt.title(f"{title_prefix}F1-score Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary_table(all_results: dict):
    rows = []
    for name, r in all_results.items():
        rows.append([name, r["AA"], r["FM"], r["BWT"]])
    df = pd.DataFrame(rows, columns=["Model", "AA", "FM", "BWT"])
    df = df.sort_values(by="AA", ascending=False).reset_index(drop=True)
    print("\n=== Summary (sorted by AA desc) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


# ------------------------------------------
# 7) Main
# ------------------------------------------
def main():
    train_path = "data/KDDTrain+.txt"
    test_path = "data/KDDTest+.txt"

    print("[1] Loading data...")
    df_train, df_test = load_nsl_kdd(train_path, test_path)

    print("[2] Preprocessing (binary + factorize categoricals)...")
    X_train, y_train = preprocess_binary(df_train)
    X_test, y_test = preprocess_binary(df_test)

    print("[3] Creating drift periods...")
    # Bạn có thể đổi schedule cho “drift mạnh/nhẹ”
    schedule = [0.0, 0.25, 0.50, 0.75, 0.90]
    periods = create_drift_periods(
        X_train, y_train, X_test, y_test,
        n_periods=5,
        period_size=8000,
        test_mix_schedule=schedule
    )
    print(f"    Created {len(periods)} periods, schedule(test_mix)={schedule}")

    print("[4] Running Static baseline...")
    static_res = run_static(periods)

    print("[5] Running ARF variants (like notebook)...")
    arf_variants = build_arf_variants()
    arf_results = run_arf_variants(periods, arf_variants)

    # Combine results
    all_results = {"Static": static_res}
    all_results.update(arf_results)

    # Print summary
    summary_df = print_summary_table(all_results)

    # Pick one ARF to compare & plot (recommend ARF_ADWIN as report main)
    best_name = "ARF_ADWIN" if "ARF_ADWIN" in all_results else summary_df.iloc[0]["Model"]
    print(f"\n[6] Plotting Static vs {best_name} ...")
    plot_compare_two(periods, static_res, all_results[best_name], title_prefix=f"{best_name} | ")

    # Also print “before/after” explicitly
    print("\n=== Before/After (Static vs chosen ARF) ===")
    before = static_res
    after = all_results[best_name]
    print(f"Static:   AA={before['AA']:.4f} | FM={before['FM']:.4f} | BWT={before['BWT']:.4f}")
    print(f"{best_name}: AA={after['AA']:.4f} | FM={after['FM']:.4f} | BWT={after['BWT']:.4f}")

    # Optional: save summary to csv
    out_csv = "results_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()
