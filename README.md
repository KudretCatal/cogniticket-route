# CogniTicket Route

[Turkish documentation](README_TR.md) · [Architecture](ARCHITECTURE.md) · [Validation](VALIDATION.md)

CogniTicket Route is a Streamlit prototype for routing Turkish smart-home support requests. It compares TF-IDF with logistic regression, a Word2Vec neural network, a BERTurk teacher, and a distilled INT8 BERT student. The INT8 student supplies the primary route. Low-confidence predictions go to a human agent. All inference runs locally on the CPU.

## Setup

Use **64-bit Python 3.12** and an **x86-64 Intel or AMD CPU**. Allow several GB of free RAM for model loading. A GPU is optional for training and unnecessary for this app.

1. Clone the repository or use **Code → Download ZIP** on GitHub and extract it. Open a terminal in the extracted folder containing `app.py`.
2. Download the following files **individually** from the [Google Drive model folder](https://drive.google.com/drive/folders/1HeyWQafwUFdofo7SJ3z1Wgub8wwRbnru). Put the five ZIP files inside the existing `archives/` folder beside `app.py`. Keep their original filenames and leave them zipped.

   - `baseline_stage2.zip`
   - `word2vec_nn_stage2.zip`
   - `bert_teacher_stage2.zip`
   - `bert_student_int8_stage2.zip`
   - `final_test_stage2.zip`

3. Install the dependencies and prepare the models.

### Windows (Command Prompt)

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe prepare_models.py
```

Launch the app:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

After setup, you can also double-click `run.bat`.

### Linux (x86-64)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-torch.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_models.py
```

Launch the app:

```bash
.venv/bin/python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) if the browser does not open automatically. Keep the terminal running. The app loads the models on the first analysis and caches them for subsequent requests.

## Try the app

| Enter this message | What to look for |
| --- | --- |
| `Kameram Wi-Fi ağına bağlanmıyor, model CAM-204` | The app extracts `CAM-204`, shows four predictions, and routes the request to Connectivity & Setup. |
| `Yarın hava nasıl?` | The INT8 student routes the out-of-scope request to a human agent. |
| `Merhaba` | A rule replies to the greeting without running model inference. |
| `Kamerra Wi-Fi ağına bağlanmıyor.` | The app suggests `kamera`. Tick the spelling correction option to apply the suggestion before classification. |

The app normalizes Unicode and whitespace, preserves case and negation, and extracts product codes with a regular expression. Optional product-name correction uses edit distance against a small vocabulary. The Transformer models accept up to 64 tokens per input. Each model uses a fixed threshold selected on validation data. The INT8 student inherits the FP32 student's threshold.

## Evaluation

The [archived results](reports/final_results.csv) come from the Colab experiment. The data consists of ChatGPT-generated Turkish examples: 120 training messages and 20 test messages, split evenly between support requests and out-of-scope messages in the test set. The deployed INT8 student achieved **80% overall routing accuracy** and **80% out-of-scope recall** on this test. English and real customer messages remain unevaluated.

The size metric counts learned weights and learned preprocessors. CPU latency averages 100 measured runs after 10 warm-ups, with preprocessing and tokenization included. The recorded timings exclude model loading and the UI. The live app adds regex normalization and optional spelling correction that were outside the archived evaluation.

## Model compatibility and verification

The training environment used PyTorch 2.14.1, Transformers 4.57.1, Gensim 4.4.0 and scikit-learn 1.6.1. The loader selects an available compatible CPU quantization engine from `x86`, `fbgemm` and `onednn`. Use the linked archives from this repository. Some model assets use pickle-based serialization.

To compare the reloaded models against the archived routing decisions, run `verify_models.py` with the virtual environment's Python after model preparation.

## AI use

Gemini assisted with the initial MASSIVE experiment and interface draft. ChatGPT helped generate the synthetic support messages and draft the revised training, distillation, quantization, evaluation and app code. The author ran the training and recorded the benchmark results in Colab.

The pretrained teacher checkpoint is [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased).
