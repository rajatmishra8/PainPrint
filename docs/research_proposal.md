# PainPrint: Chronic Pain Prediction from Passive Digital Behavior
## Masters Research Proposal

**Author:** Rajat Mishra  
**Degree:** Master of Science in Data Science  
**Submission Date:** 2024  

---

## 1. Introduction & Motivation

Chronic pain affects over 1.5 billion people globally and is among the leading causes of disability. Current clinical assessment relies almost entirely on subjective self-reporting — patients rate their pain on a numerical scale during infrequent clinical visits. This approach suffers from:

- **Recall bias**: patients cannot accurately remember pain levels from days ago
- **Effort burden**: daily diary compliance drops below 40% within two weeks
- **Temporal sparsity**: data captured only at appointments misses day-to-day variation
- **Subjectivity**: no two patients calibrate the same scale identically

Smartphones are used by over 6 billion people and passively capture rich behavioral signals throughout the day. We hypothesize that **chronic pain measurably and consistently alters smartphone interaction patterns**, creating a *Digital Pain Signature* unique to each individual.

---

## 2. Research Questions

1. Can passive smartphone behavioral features predict same-day pain scores with clinically meaningful accuracy (MAE < 1.5 on a 0–10 scale)?
2. Does each user exhibit a stable "Digital Pain Signature" that differentiates them from population-level averages?
3. Can federated learning preserve predictive accuracy while ensuring complete data privacy?
4. Can intra-day pain spikes be detected without any user input?

---

## 3. Novel Contributions

### 3.1 Digital Pain Signature (DPS)
We introduce DPS: a user-specific behavioral fingerprint computed as per-feature z-scores relative to the individual's own rolling baseline. Unlike population-level biomarkers, DPS captures *change from self* rather than *deviation from average*, making it robust to inter-user behavioral heterogeneity.

### 3.2 Cultural Pain Half-Life (CPH)
Inspired by Ebbinghaus forgetting curves, CPH models how quickly a user's behavioral patterns "recover" after a pain flare. Users with shorter CPH may have better pain coping mechanisms. This is the first application of forgetting-curve mathematics to pain behavior.

### 3.3 Privacy-Preserving Architecture
Using Federated Averaging (FedAvg), PainPrint trains a shared LSTM across user devices without transmitting raw behavioral data. Each device trains locally; only gradient updates are aggregated on the server.

---

## 4. Methodology

### 4.1 Data Collection
**Target Population:** Adults (18–65) with diagnosed chronic conditions (fibromyalgia, rheumatoid arthritis, chronic back pain).

**Passive Features Captured:**
| Feature | Proxy For |
|---|---|
| Typing speed (WPM) | Motor control degradation |
| Backspace rate | Cognitive load / concentration |
| Inter-key delay | Fine motor disruption |
| Scroll velocity | Physical ease of movement |
| App switch rate | Restlessness / discomfort |
| Session duration | Sustained attention capacity |
| Night screen usage | Pain-induced insomnia |
| Touch pressure | Grip strength variation |
| Notification response delay | Motivation / energy |

**Ground Truth:** Daily pain score via a single slider notification (0–10). Takes < 3 seconds.

### 4.2 Feature Engineering
- Rolling 3-hour window statistics (mean, std)
- Circadian encoding (hour → sin/cos)
- Digital Pain Signature (per-user z-score normalization)
- Day-of-week embeddings

### 4.3 Model Architecture

```
Input: Sequence of 16 hours × 42 features
    ↓
LSTM Layer 1 (128 units) + Dropout(0.3)
    ↓
LSTM Layer 2 (64 units) + Dropout(0.2)
    ↓
Dense(32) → ReLU → Dense(1)
    ↓
Output: Predicted Pain Score (0–10)
```

### 4.4 Federated Learning Protocol
- **Algorithm:** FedAvg (McMahan et al., 2017)
- **Rounds:** 20 global rounds
- **Clients per round:** 10 randomly sampled users
- **Local epochs:** 3 per round
- **Differential privacy:** Gaussian noise injection (ε = 1.0)

### 4.5 Evaluation Metrics
- Mean Absolute Error (MAE) — primary metric
- Root Mean Square Error (RMSE)
- Pearson correlation (r) between predicted and actual
- Clinically significant threshold: MAE < 1.5 (validated in pain literature)

---

## 5. Ethical Considerations

- IRB approval required for human subject data collection
- Explicit informed consent for all behavioral logging
- Right to data deletion at any time
- Federated architecture ensures no raw data transmission
- Study does not replace clinical care — supplementary only

---

## 6. Expected Outcomes

| Metric | Target |
|---|---|
| MAE (centralized) | < 1.2 |
| MAE (federated) | < 1.5 |
| R² | > 0.65 |
| Intra-day spike detection AUC | > 0.80 |

---

## 7. Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Literature Review | 4 weeks | Annotated bibliography |
| Data Pipeline + Simulator | 6 weeks | Working data generator |
| Feature Engineering | 4 weeks | Feature importance analysis |
| LSTM Model Training | 6 weeks | Centralized model + results |
| Federated Learning | 6 weeks | FL simulation + comparison |
| Dashboard | 4 weeks | Interactive Dash app |
| Thesis Writing | 8 weeks | Full thesis document |

---

## 8. Publications Roadmap

1. **"Digital Pain Signatures: Passive Behavioral Biomarkers for Chronic Pain Assessment"** — target: *npj Digital Medicine*
2. **"Privacy-Preserving Pain Prediction via Federated Mobile Sensing"** — target: *JAMIA*
3. **"Intra-day Pain Spike Detection Without Self-Reporting"** — target: *ICDH 2025*

---

## 9. References

1. Torous, J. et al. (2018). New Tools for New Research in Psychiatry. *JMIR Mental Health*.
2. McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks. *AISTATS*.
3. Pratap, A. et al. (2020). Indicators of Retention in Remote Digital Health Studies. *npj Digital Medicine*.
4. Dworkin, R.H. et al. (2008). Interpreting the Clinical Importance of Treatment Outcomes. *Journal of Pain*.
5. Ebbinghaus, H. (1885). *Über das Gedächtnis*. Leipzig: Duncker & Humblot.

---

*Rajat Mishra | Masters in Data Science*  
*"Pain leaves no fingerprint — until now."*
