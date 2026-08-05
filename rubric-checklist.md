# ✅ DDM501 Lab 2 — Step-by-Step Rubric Checklist

> Dùng bảng này để **tự kiểm tra** trước khi nộp bài.  
> Tổng điểm: **100%** = Pipeline Quality (35%) + Experiment Tracking (25%) + Airflow Automation (20%) + Documentation (20%)

> [!IMPORTANT]
> Project có **2 virtual environments** khác nhau:
> - `venv/` — dùng cho ML pipeline (mlflow, surprise, ...)
> - `venv-airflow/` — dùng cho Airflow DAG
>
> Khi chạy lệnh liên quan đến Airflow, phải dùng `venv-airflow/bin/python` thay vì `python` thông thường!

---

## 🏗️ PHẦN 1 — Pipeline Quality (35%)

### 1A · Modular Structure (10%)

Kiểm tra xem pipeline có được tách module rõ ràng không.

**✔ Những gì cần có:**

- [ ] `pipeline/data_ingestion.py` — tách riêng logic load data
- [ ] `pipeline/preprocessing.py` — tách riêng logic validate/preprocess
- [ ] `pipeline/training.py` — tách riêng logic train + MLflow log
- [ ] `pipeline/evaluation.py` — tách riêng logic evaluate + metrics
- [ ] `pipeline/registry.py` — tách riêng logic register model

**🔍 Cách kiểm tra:**

```bash
# Mỗi file phải có thể chạy độc lập (không crash khi import)
cd ddm501-lab2-starter
source venv/bin/activate
python -c "from pipeline import data_ingestion, preprocessing, training, evaluation, registry; print('OK')"
```

Nếu không có error → ✅ đạt modular.

---

### 1B · Reproducible Execution (10%)

Pipeline phải cho cùng kết quả nếu chạy lại với cùng config.

**✔ Những gì cần có:**

- [ ] `random_state = 42` được set nhất quán trong config
- [ ] `test_size = 0.2` cố định
- [ ] SVD/NMF dùng `random_state` khi khởi tạo model

**🔍 Cách kiểm tra:**

```bash
# Chạy pipeline 2 lần, so sánh RMSE
python -m pipeline.run_pipeline --model-type svd 2>&1 | grep "RMSE"
python -m pipeline.run_pipeline --model-type svd 2>&1 | grep "RMSE"
```

Hai lần cho cùng RMSE → ✅ reproducible.

> **Kiểm tra code:** Mở [training.py](file:///Users/hd-569/Downloads/DDM501_Assigments_Labs_Projects/Labs/Lab01.2/ddm501-lab2-starter/pipeline/training.py#L60-L60) → dòng `_SEEDED_MODELS = {"svd", "nmf"}` + `params.setdefault("random_state", RANDOM_STATE)` ✅ đã có.

---

### 1C · Error Handling (8%)

Code phải xử lý lỗi đúng cách, không crash thầm lặng.

**✔ Những gì cần có:**

- [ ] `train_model()` raise `ValueError` nếu `trainset is None`
- [ ] `evaluate_model()` raise `ValueError` nếu `model is None` hoặc `testset` rỗng
- [ ] MLflow server không kết nối được → fallback về local file store (không crash)
- [ ] Training fail → tag `training_status = failed` trong MLflow run

**🔍 Cách kiểm tra:**

```bash
python -c "
from pipeline.training import train_model
try:
    train_model(None, 'svd')
except ValueError as e:
    print('✅ ValueError caught:', e)
"
```

```bash
python -c "
from pipeline.evaluation import evaluate_model
try:
    evaluate_model(None, [], run_id=None, log_to_mlflow=False)
except ValueError as e:
    print('✅ ValueError caught:', e)
"
```

---

### 1D · Code Quality (7%)

**✔ Những gì cần có:**

- [ ] Có docstrings cho tất cả public functions
- [ ] Type hints đầy đủ (`-> Dict[str, Any]`, v.v.)
- [ ] Logging thay vì `print()` trong pipeline code
- [ ] Không có code "dead" (commented out, unused imports)

**🔍 Cách kiểm tra:**

```bash
# Kiểm tra docstrings
python -c "
import pipeline.training as t
fns = [t.train_model, t.build_model, t.setup_mlflow, t.flatten_params]
for f in fns:
    status = '✅' if f.__doc__ else '❌ MISSING'
    print(f'{status} {f.__name__}')
"
```

---

## 📊 PHẦN 2 — Experiment Tracking (25%)

### 2A · MLflow Setup Correct (8%)

**✔ Những gì cần có:**

- [ ] MLflow server đang chạy (port 5000) HOẶC có fallback local
- [ ] Experiment name được set (`movie-rating-prediction`)
- [ ] `setup_mlflow()` được gọi trước khi train

**🔍 Cách kiểm tra:**

```bash
# Start MLflow server (nếu chưa chạy)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

# Truy cập UI
open http://localhost:5000
```

Vào UI, tìm experiment **"movie-rating-prediction"** → ✅ setup đúng.

---

### 2B · Parameters Logged (5%)

**✔ Những gì cần có trong mỗi MLflow run:**

- [ ] `model_type` (svd / nmf / knn)
- [ ] Hyperparameters: `n_factors`, `n_epochs`, `lr_all`, v.v.
- [ ] Dataset context: `dataset`, `test_size`, `random_state`
- [ ] Data stats: `n_train_users`, `n_train_items`, `n_train_ratings`

**🔍 Cách kiểm tra:**

1. Vào http://localhost:5000
2. Click vào experiment **"movie-rating-prediction"**
3. Click vào một run bất kỳ
4. Tab **Parameters** → kiểm tra có đủ các params trên không

Hoặc kiểm tra qua code:

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('movie-rating-prediction')
if exp:
    runs = client.search_runs([exp.experiment_id], max_results=1)
    if runs:
        print('Params:', list(runs[0].data.params.keys()))
"
```

---

### 2C · Metrics Logged (5%)

**✔ Những gì cần có:**

- [ ] `rmse`
- [ ] `mae`
- [ ] `mse`
- [ ] `mape`
- [ ] `coverage`
- [ ] `training_time_seconds`
- [ ] `n_predictions`, `n_impossible`

**🔍 Cách kiểm tra:**

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('movie-rating-prediction')
if exp:
    runs = client.search_runs([exp.experiment_id], max_results=1)
    if runs:
        metrics = list(runs[0].data.metrics.keys())
        required = ['rmse','mae','mse','coverage','training_time_seconds']
        for m in required:
            status = '✅' if m in metrics else '❌ MISSING'
            print(f'{status} {m}')
"
```

---

### 2D · Artifacts Logged (4%)

**✔ Những gì cần có:**

- [ ] Model pickle (`pickle/model_svd.pkl`)
- [ ] MLflow pyfunc model (`model/` folder)
- [ ] Plot: `plots/prediction_distribution.png`
- [ ] Plot: `plots/error_by_rating.png`
- [ ] Text report: `reports/evaluation_report_*.txt`

**🔍 Cách kiểm tra:**

1. Vào http://localhost:5000
2. Click vào một run đã hoàn thành
3. Tab **Artifacts** → kiểm tra có đủ các folders/files trên

---

### 2E · Model Registry (3%)

**✔ Những gì cần có:**

- [ ] Có ít nhất 1 model được register trong Model Registry
- [ ] Model có stage **"Production"**

**🔍 Cách kiểm tra:**

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
from pipeline.registry import list_registered_models
models = list_registered_models()
if models:
    print('✅ Registered models:')
    for m in models:
        print(f'  - {m[\"name\"]}:', m['latest_versions'])
else:
    print('❌ No registered models found')
"
```

Hoặc vào UI: tab **Models** ở menu trên → kiểm tra.

---

## 🔄 PHẦN 3 — Airflow Automation (20%)

### 3A · DAG Structure Correct (8%)

**✔ Những gì cần có:**

- [ ] DAG ID: `movie_rating_training`
- [ ] Đủ 7 tasks: `load_data → preprocess_data → train_model → evaluate_model → decide_registration → [register_model | skip_registration] → cleanup`
- [ ] Sử dụng `BranchPythonOperator` cho quality gate
- [ ] `trigger_rule='none_failed'` cho cleanup task
- [ ] `max_active_runs=1` (no overlap)
- [ ] `catchup=False`

**🔍 Cách kiểm tra (không cần Airflow chạy):**

```bash
# ⚠️ Phải dùng venv-airflow, không phải venv!
export AIRFLOW_HOME=$(pwd)/airflow_home
venv-airflow/bin/python -c "
import sys, os
sys.path.insert(0, '.')
from dags.ml_training_dag import dag
print('✅ DAG ID:', dag.dag_id)
print('✅ Tasks:', [t.task_id for t in dag.tasks])
print('✅ Schedule:', dag.schedule_interval)
print('✅ Max active runs:', dag.max_active_runs)
print('✅ Catchup:', dag.catchup)
"
```

**Kết quả đã verify ✅:**
```
✅ DAG ID: movie_rating_training
✅ Tasks: ['load_data', 'preprocess_data', 'train_model', 'evaluate_model', 'decide_registration', 'register_model', 'skip_registration', 'cleanup']
✅ Schedule: @weekly
✅ Max active runs: 1
✅ Catchup: False
```

---

### 3B · Tasks Execute Properly (7%)

**✔ Những gì cần có:**

- [ ] Mỗi task function gọi đúng pipeline module tương ứng
- [ ] XCom được dùng để pass data giữa tasks (run_id, metrics, data_path)
- [ ] Branching logic hoạt động: chỉ register khi RMSE < threshold

**🔍 Cách kiểm tra:**

```bash
# ⚠️ Phải dùng venv-airflow!
export AIRFLOW_HOME=$(pwd)/airflow_home
venv-airflow/bin/airflow db migrate 2>/dev/null
venv-airflow/bin/airflow dags test movie_rating_training 2024-01-07 2>&1 | tail -20
```

Nếu thấy "Marking task as SUCCESS" cho tất cả tasks → ✅

> **Kiểm tra nhanh không cần chạy DAG:** Dùng lệnh ở 3A — nếu parse được = tasks đúng ✅

---

### 3C · Schedule Configured (5%)

**✔ Những gì cần có:**

- [ ] Schedule = `@weekly` (hoặc cron `0 0 * * 0` = mỗi Chủ nhật 00:00)

**🔍 Cách kiểm tra:**

```bash
# Dùng venv thường (pipeline/config.py không cần airflow)
source venv/bin/activate
python -c "
from pipeline.config import AIRFLOW_SCHEDULE
print('Schedule:', AIRFLOW_SCHEDULE)
"
```

**Kết quả đã verify ✅:** `Schedule: @weekly`

---

## 📝 PHẦN 4 — Documentation (20%)

### 4A · Experiment Report (10%)

**✔ Những gì cần có trong `experiment_report.md`:**

- [ ] Ít nhất **5 experiments** được so sánh (thực tế code có 9 ✅)
- [ ] Bảng kết quả với RMSE, MAE của từng config
- [ ] Phân tích theo model family (SVD vs NMF vs KNN)
- [ ] Kết luận: model nào tốt nhất, tại sao
- [ ] Biểu đồ so sánh (artifacts/experiment_comparison.png)
- [ ] Recommendation model nào nên promote lên Production

**🔍 Cách kiểm tra:**

```bash
# Xem report
cat experiment_report.md
# Kiểm tra biểu đồ tồn tại
ls artifacts/experiment_comparison.png && echo "✅ Chart exists" || echo "❌ Chart missing"
```

---

### 4B · README Complete (5%)

**✔ Những gì cần có trong `README.md`:**

- [ ] Mô tả project overview
- [ ] Hướng dẫn setup (venv + pip install)
- [ ] Cách start MLflow server
- [ ] Cách chạy pipeline (`python -m pipeline.run_pipeline`)
- [ ] Cách chạy experiment sweep
- [ ] Cách test DAG Airflow
- [ ] Bảng configuration reference
- [ ] Grading rubric (bonus)

**🔍 Cách kiểm tra:**

```bash
wc -l README.md  # Phải > 100 dòng
grep -c "##" README.md  # Phải có nhiều sections
```

---

### 4C · Code Documentation (5%)

**✔ Những gì cần có:**

- [ ] Module-level docstring trong mỗi file (ở đầu file)
- [ ] Function-level docstring với Args, Returns, Raises
- [ ] Inline comments cho logic phức tạp
- [ ] Type hints

**🔍 Cách kiểm tra:**

```bash
python -c "
import pipeline.training as t
import pipeline.evaluation as e
import pipeline.registry as r
for mod in [t, e, r]:
    status = '✅' if mod.__doc__ else '❌ MISSING'
    print(f'{status} {mod.__name__} module docstring')
"
```

---

## 🎯 ĐIỂM SUMMARY

Chạy lệnh này để kiểm tra nhanh tất cả (đã test và verify ✅):

```bash
cd ddm501-lab2-starter

echo "=== 1. Module imports (venv) ==="
venv/bin/python -c "from pipeline import data_ingestion, preprocessing, training, evaluation, registry; print('✅ All modules OK')"

echo "=== 2. Error handling ==="
venv/bin/python -c "
from pipeline.training import train_model
try:
    train_model(None, 'svd')
    print('❌ Should have raised ValueError')
except ValueError:
    print('✅ train_model error handling OK')
"
venv/bin/python -c "
from pipeline.evaluation import evaluate_model
try:
    evaluate_model(None, [], run_id=None, log_to_mlflow=False)
except ValueError:
    print('✅ evaluate_model error handling OK')
"

echo "=== 3. DAG structure (venv-airflow) ==="
export AIRFLOW_HOME=$(pwd)/airflow_home
venv-airflow/bin/python -c "
import sys; sys.path.insert(0, '.')
from dags.ml_training_dag import dag
tasks = [t.task_id for t in dag.tasks]
expected = ['load_data','preprocess_data','train_model','evaluate_model','decide_registration','register_model','skip_registration','cleanup']
for t in expected:
    status = '✅' if t in tasks else '❌ MISSING'
    print(f'{status} Task: {t}')
print('✅ Schedule:', dag.schedule_interval)
print('✅ Max active runs:', dag.max_active_runs)
print('✅ Catchup:', dag.catchup)
"

echo "=== 4. Experiment report ==="
venv/bin/python -c "
import re, os
with open('experiment_report.md') as f:
    content = f.read()
count = len(re.findall(r'\| \d+ \|', content))
status = '✅' if count >= 5 else '❌ Need >= 5'
print(f'{status} {count} experiments in report')
chart = os.path.exists('artifacts/experiment_comparison.png')
print('✅ Chart exists' if chart else '❌ Chart missing')
"

echo "=== 5. README sections ==="
venv/bin/python -c "
with open('README.md') as f:
    content = f.read()
sections = ['Quick Start', 'MLflow', 'Airflow', 'Experiment', 'Grading']
for s in sections:
    status = '✅' if s in content else '❌ MISSING'
    print(f'{status} Section: {s}')
"

echo "=== 6. Docstrings ==="
venv/bin/python -c "
import pipeline.training as t
import pipeline.evaluation as e
import pipeline.registry as r
for mod in [t, e, r]:
    status = '✅' if mod.__doc__ else '❌ MISSING'
    print(f'{status} {mod.__name__} module docstring')
"

echo "=== 7. MLflow local runs ==="
venv/bin/python -c "
import os
exps = [d for d in os.listdir('mlruns') if os.path.isdir(f'mlruns/{d}') and d != '.trash']
print(f'✅ mlruns/ has {len(exps)} experiment(s)')
"
```

**Kết quả đã chạy thực tế:**

```
=== 1. Module imports === ✅ All modules OK
=== 2. Error handling  === ✅ train_model / evaluate_model OK
=== 3. DAG structure   === ✅ 8/8 tasks  |  @weekly  |  max_runs=1  |  catchup=False
=== 4. Experiment      === ✅ 12 experiments  |  ✅ Chart exists
=== 5. README          === ✅ All sections present
=== 6. Docstrings      === ✅ training / evaluation / registry
=== 7. MLflow runs     === ✅ 5 experiments in mlruns/
```

---

## ⚠️ NHỮNG THỨ CÒN THIẾU (cần làm trước khi nộp)

Theo README checklist:

- [ ] **Screenshots của MLflow UI** — bắt buộc theo submission requirements!
  - Screenshot trang danh sách experiments
  - Screenshot compare runs (sort by RMSE)
  - Screenshot Model Registry tab
  - Screenshot artifacts của một run
- [ ] **GitHub repository link** — submit link repo lên

**Cách lấy screenshots:**

```bash
# Start MLflow server nếu chưa chạy
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Mở browser
open http://localhost:5000
```

Chụp màn hình các tab: **Experiments**, **Compare Runs**, **Models** (Registry).

---

## 📋 RUBRIC SCORE ESTIMATE

| Tiêu chí                | Max      | Đánh giá                            |
| ----------------------- | -------- | ----------------------------------- |
| Modular structure       | 10%      | ✅ 5 files tách riêng rõ ràng       |
| Reproducible execution  | 10%      | ✅ random_state=42 cố định          |
| Error handling          | 8%       | ✅ ValueError, fallback, tags       |
| Code quality            | 7%       | ✅ Docstrings, type hints, logging  |
| **Pipeline Quality**    | **35%**  | **~33-35%**                         |
| MLflow setup            | 8%       | ✅ server + experiment config       |
| Parameters logged       | 5%       | ✅ model params + data context      |
| Metrics logged          | 5%       | ✅ RMSE/MAE/MSE/coverage/...        |
| Artifacts logged        | 4%       | ✅ pickle + pyfunc + plots + report |
| Model registry          | 3%       | ✅ register + stage transition      |
| **Experiment Tracking** | **25%**  | **~23-25%**                         |
| DAG structure           | 8%       | ✅ 7 tasks + branching              |
| Tasks execute properly  | 7%       | ✅ XCom + proper callables          |
| Schedule configured     | 5%       | ✅ @weekly                          |
| **Airflow Automation**  | **20%**  | **~18-20%**                         |
| Experiment report       | 10%      | ✅ 9 experiments + analysis         |
| README complete         | 5%       | ✅ full setup guide                 |
| Code documentation      | 5%       | ✅ docstrings + type hints          |
| **Documentation**       | **20%**  | **~18-20%**                         |
| **TỔNG**                | **100%** | **~92-100%** ⭐                     |

> [!IMPORTANT]
> Điểm có thể trừ nếu **thiếu screenshots MLflow UI** — đây là yêu cầu bắt buộc trong phần 5.3 Submission!
