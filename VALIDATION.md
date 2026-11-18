# Validation

## Local checks on November 15, 2026

The Linux x86-64 CPU environment used Python 3.12, PyTorch 2.14.1+cpu, Transformers 4.57.1, Gensim 4.4.0, scikit-learn 1.6.1, and Streamlit 1.51.0.

- All five ZIP archives were extracted, and all four live models loaded from local files without downloading model weights.
- The reload check compared four live models against the archived decisions on 20 raw test inputs. All 80 final routing decisions matched. The additional FP32 student benchmark row was outside this check. The check reused archived inputs and left the original test results intact.
- One rejected INT8 message received a different raw category. Both predictions routed it to a human agent. The archived run used a batch size of 4, and the app uses a batch size of 1. Batch size and hardware can affect dynamic INT8 outputs. The weights and threshold used for the check were unchanged.
- Streamlit AppTest covered initial rendering, the empty-input warning, the rule-based reply to “Merhaba,” and inference with all four models for “Kameram Wi-Fi ağına bağlanmıyor, model CAM-204.” The INT8 student routed that message to Connectivity & Setup.
- Additional checks covered model-code extraction, whitespace normalization, preservation of negation, product-name spelling suggestions, and classification of a support request that begins with a greeting. The Python source files compiled successfully.

The local checks used Streamlit's test runtime. Browser-based visual review was unavailable in that environment because the headless browser download failed.

## Follow-up checks

The author ran the app on Windows using the `onednn` backend and checked support routing, out-of-scope fallback, the greeting rule, and optional spelling correction.

On November 18, the current repository code was checked in a separate Python 3.12 environment on Linux x86-64. The archive preparation script accepted all five original ZIP files. Streamlit AppTest showed all four predictions and the results table. It also confirmed that the camera connection request routed to Connectivity & Setup and “Yarın hava nasıl?” routed to a human agent.

See the [archived reload check](reports/reload_verification.json) and [test results](reports/final_results.csv) for the recorded evidence.
