# PPG 呼吸速率估算

從光體積變化描記訊號(PPG,也就是血氧機與智慧手錶上的光學脈搏訊號)估算呼吸速率,比較傳統訊號處理和機器學習兩類方法,並提供一個可以用手機鏡頭即時量測的網頁展示。

> 作品集技術展示,未經臨床驗證,不可用於診斷或病人監測。

## 為什麼做這個

呼吸速率是預警病情惡化最敏感的生命徵象之一(NEWS 等早期預警評分都包含它),但在病房裡最常被目測估算,甚至漏量。血氧機本來就有 PPG 訊號,如果能從同一個訊號估算呼吸速率,就不需要額外的感測器。

## 方法

呼吸會在 PPG 上留下三種調變(Karlen et al., 2013):

| 調變 | 生理機制 | 程式中的名稱 |
|---|---|---|
| 基線飄移 | 胸腔壓力改變靜脈回流 | `riiv` |
| 脈搏振幅變化 | 每搏輸出量隨呼吸改變 | `riav` |
| 心跳間隔變化 | 呼吸性竇性心律不整 | `rifv` |

處理流程(`src/ppgrr/dsp.py`):

1. PPG 濾波(0.5–8 Hz)後偵測每一下心跳的波峰和波谷。
2. 每下心跳取一個值,形成三條逐拍序列,重新取樣到 4 Hz。
3. 在呼吸頻帶(每分鐘 4–42 次)內用頻譜峰值和呼吸計數兩種方式估算呼吸速率。
4. **Smart fusion:** 三種估算的標準差 ≤ 4 次/分時才輸出平均值,否則判定為訊號不可靠並拒絕輸出。
5. **LightGBM:** 以上述估算值和訊號品質特徵(心跳數、心跳間隔變異、PPG 偏度與峰度、頻譜集中度)作為輸入,學習修正誤差。

## 資料

[BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/)(PhysioNet,ODC-BY 1.0):53 位 ICU 病人,每位 8 分鐘,PPG 取樣率 125 Hz。參考呼吸速率由兩位標註者在阻抗呼吸訊號上逐次標註呼吸。

- 視窗:32 秒,每 4 秒滑動一次。
- 參考值:視窗內呼吸間隔平均值換算的呼吸速率(兩位標註者取平均)。
- **依病人分組的 5 折交叉驗證:** 每位病人的預測都來自沒看過這位病人的模型,避免同一病人的視窗同時出現在訓練和測試資料中。

## 結果

53 位病人、7,518 個視窗,依病人分組 5 折交叉驗證(`results/metrics.csv`):

| 方法 | 涵蓋率 | MAE(次/分) | 偏差 | 95% 一致性界限 | 誤差 ≤ 2 次/分 |
|---|---|---|---|---|---|
| 基線飄移(頻譜) | 100% | 2.47 | −1.57 | −9.2 ~ +6.1 | 67% |
| 脈搏振幅(頻譜) | 100% | 4.52 | −2.17 | −13.5 ~ +9.2 | 41% |
| 心跳間隔(頻譜) | 100% | 4.78 | +0.90 | −12.3 ~ +14.1 | 43% |
| Smart fusion(頻譜) | 64% | 2.57 | −1.73 | −7.9 ~ +4.4 | 52% |
| Smart fusion(呼吸計數) | 85% | 2.06 | −0.36 | −5.6 ~ +4.8 | 57% |
| **LightGBM** | 100% | **1.49** | +0.04 | −4.3 ~ +4.4 | **75%** |
| LightGBM(僅 fusion 接受的視窗) | 64% | 1.36 | +0.04 | −4.0 ~ +4.1 | 77% |
| 兩位標註者之間(上限參考) | 100% | 0.32 | +0.12 | −1.7 ~ +2.0 | 94% |

![Bland-Altman](results/bland_altman.png)

**重要發現:LightGBM 會往平均值靠攏。** 資料中 95% 的視窗呼吸速率落在每分鐘 10–20 次,模型在這個範圍表現最好,但在極端值反而比單一訊號處理方法差(`results/mae_by_rr_band.csv`):

| 參考呼吸速率 | 基線飄移 MAE | LightGBM MAE | 視窗數 |
|---|---|---|---|
| < 12 次/分 | 1.56 | 2.97 | 少數 |
| 12–20 次/分 | 2.67 | 1.11 | 多數 |
| 20–30 次/分 | 2.61 | 7.19 | 43 |

呼吸過快(> 20 次/分)正是臨床上最需要抓到的情況,所以平均誤差最低,不代表臨床上最有用。改善方向:依呼吸速率分層加權、蒐集更多高呼吸速率的資料,或讓模型輸出不確定性,在把握不足時拒絕輸出。

## 專案結構

```
src/ppgrr/      dsp.py(訊號處理)、data.py(資料載入)、features.py、evaluate.py
scripts/        prepare_data.py → run_experiments.py → export_demo.py
tests/          合成訊號的單元測試
web/            手機鏡頭即時量測與資料集重播(純前端,ppg-dsp.js 是 dsp.py 的 JavaScript 版本)
```

## 重現步驟

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
# 下載 https://physionet.org/content/bidmc/get-zip/1.0.0/ 並解壓縮到 data/raw/
python scripts/prepare_data.py
python scripts/run_experiments.py
python scripts/export_demo.py
python -m http.server 8000 -d web   # 開啟 http://localhost:8000
```

## 限制

- 資料只有 53 位病人,來自單一醫院的 ICU,而且都是靜止狀態;沒有測試走動或運動時的表現。
- 參考值來自阻抗呼吸訊號的人工標註,本身也有誤差(見結果中兩位標註者之間的一致性)。
- 尚未做外部驗證。下一步可以用 CapnoBase(以二氧化碳監測為參考值)作為獨立測試集。
- 手機鏡頭量測的訊號品質比醫療級血氧機差,網頁展示只用來說明原理。

## 參考文獻

- Karlen W, et al. Multiparameter respiratory rate estimation from the photoplethysmogram. *IEEE Trans Biomed Eng* 2013;60:1946–53.
- Pimentel MAF, et al. Toward a robust estimation of respiratory rate from pulse oximeters. *IEEE Trans Biomed Eng* 2017;64:1914–23.
- Charlton PH, et al. Breathing rate estimation from the electrocardiogram and photoplethysmogram: a review. *IEEE Rev Biomed Eng* 2018;11:2–20.
- Schäfer A, Kratky KW. Estimation of breathing rate from respiratory sinus arrhythmia. *Ann Biomed Eng* 2008;36:476–85.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 2000;101:e215–20.
