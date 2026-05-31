# Credit Risk Modeling Project

## Credit Scoring Business Understanding

This section explores the structural, regulatory, and financial mechanics governing the development and deployment of credit risk scorecards, directly aligning our technical implementation with institutional standards.

---

### 1. Basel II Accord & The Mandate for Model Interpretability

The Basel II Accord (and its subsequent evolutions under Basel III/IV) established a global framework for financial institutions to calculate **Regulatory Capital Requirements** based on three core pillars:
1. Minimum Capital Requirements (based on risk metrics)
2. Supervisory Review
3. Market Discipline

Under the **Internal Ratings-Based (IRB)** approach, banks are permitted to use their own internal statistical models to estimate key risk parameters:
* **Probability of Default (PD):** The likelihood that a borrower will fail to repay over a given time horizon.
* **Loss Given Default (LGD):** The share of an asset that is lost if a borrower defaults.
* **Exposure at Default (EAD):** The total gross exposure to a facility when default occurs.

#### The Influence on Interpretability and Documentation
Because internal models directly dictate the amount of cash a bank must hold in reserve (capital cushions) rather than deploying it for interest-generating loans, regulatory bodies require absolute transparency. 

* **The "Black Box" Restriction:** Regulators do not permit models whose decision boundaries cannot be auditably traced. If a model predicts a high PD for a segment of borrowers, risk managers must be able to explain *why* (e.g., high debt ratio, low recency of transaction).
* **The Right to Explanation:** Consumer protection laws (such as the Fair Credit Reporting Act in the US or GDPR in Europe) require that applicants denied credit must be provided with specific "adverse action codes" (e.g., "revolving credit utilization too high"). 
* **Model Validation and Stress Testing:** Independent internal model validation teams and external supervisors must reproduce the model's performance under simulated economic downturns. A thoroughly documented pipeline—mapping every raw transaction to its transformed state—is legally mandatory to prevent structural model risk and multi-million dollar regulatory fines.

---

### 2. Proxy Variable Mechanics & Inherent Business Risks

In real-world transaction data or alternative data streams (such as those emphasized by the Hong Kong Monetary Authority framework), a clear, historical "Default" column rarely exists out of the box. Credit risk analysts must define a **Proxy Variable** to act as the ground truth target ($Y \in \{0, 1\}$) for machine learning models.

#### Why a Proxy Variable is Necessary
"Default" is legally and operationally defined (often under Basel guidelines) as an event where a borrower is **90+ Days Past Due (DPD)** on an obligation, or undergoes bankruptcy/restructuring. However, when parsing alternative data sources (like POS merchant cash flows or bank card transactions), an explicit credit contract tracking DPD may not be present. Analysts instead construct a proxy target, such as:
* *A merchant whose monthly transaction volume drops by more than 80% over a rolling 60-day period.*
* *An individual whose account balance drops below a critical threshold while simultaneously showing zero inbound salary indicators for two consecutive months.*

#### Business Risks of Proxy-Based Prediction

```
[ True Underlying Credit Risk ] <─── Systemic Gap ───> [ Created Proxy Target Variable ]
              │                                                     │
              ▼                                                     ▼
     Actual Default Event                                    Engineered Label
  (90+ DPD / Bankruptcy / Legal)                     (e.g., Volatility / Balances / Drops)
```

Using a proxy variable introduces critical business vulnerabilities due to the variance between the proxy and actual credit default:

* **Type I Error Inflation (False Positives / Opportunity Cost):** The model flags a borrower as a default risk based on the proxy (e.g., a massive transaction drop). However, the borrower might simply have switched processing vendors or transitioned to cash operations. The bank unnecessarily rejects a creditworthy merchant, driving away profitable business and damaging customer relationships.
* **Type II Error Inflation (False Negatives / Severe Financial Loss):** The model fails to flag a borrower because their transaction patterns appear stable according to the proxy metrics. However, behind the scenes, the business is accumulating unmonitored external debt or is structurally insolvent. The bank approves the loan, resulting in a toxic asset, a complete write-off of principal, and direct damage to the bank's non-performing loan (NPL) ratio.
* **Proxy Drift:** Behavioral habits shift over time due to economic cycles or structural market changes (e.g., macro inflation, changes in consumer spending platforms). A proxy variable calculated on 2024 transaction patterns may completely decouple from actual default rates by 2026, leading to quiet, unmonitored model degradation.

---

### 3. The Regulated Trade-Off: Logistic Regression (with WoE) vs. Gradient Boosting

Deploying credit scoring solutions in institutional environments requires balancing statistical power against operational and regulatory constraints.

| Dimension | Traditional: Logistic Regression with Weight of Evidence (WoE) | Advanced: Gradient Boosting Machines (e.g., XGBoost) |
| :--- | :--- | :--- |
| **Mathematical Structure** | Linear combination of binned variables: <br><span class="math">ln(p / (1-p)) = β₀ + β₁X₁ + ... + βₙXₙ</span> | Non-linear ensemble of sequentially optimized decision trees. |
| **Predictive Performance** | **Moderate.** Struggles to natively map complex, non-linear feature interactions or subtle step-functions without manually intensive engineering. | **High.** Natively captures high-order non-linear interactions and cross-feature relationships, minimizing structural bias and maximizing Gini/AUC scores. |
| **Interpretability & Transparency** | **Absolute.** Every input variable is binned into groups where each bin is assigned a transparent **WoE value**. This maps directly to a static, readable **Credit Scorecard** where points are added or subtracted based on clear, explicit cut-offs. | **Low / Visual Approximations Only.** Interactions are deeply embedded across hundreds of trees. Model explanations rely on post-hoc methods like SHAP or feature importances, which demonstrate correlation but do not guarantee strict, traceable causality. |
| **Regulatory Compliance & Auditing** | **Seamless.** Instantly meets Basel IRB requirements. Easy to validate, audit, and explain to non-technical stakeholders, credit officers, and central bank compliance checkers. | **Highly Restricted.** Often blocked or subjected to extreme scrutiny by model validation boards due to the risk of "unstable" or unmapped decision boundaries under extreme out-of-distribution inputs. |
| **Handling Monotonicity** | **Guaranteed.** By enforcing strict monotonic constraints on the WoE values during the binning phase, the analyst ensures that as risk indicators rise (e.g., DPD increases), the credit score *always* monotonically drops. | **Requires Manual Overrides.** Without explicitly configuring strict monotonic constraints during hyperparameter tuning, the algorithm may find local anomalies where higher debt temporarily yields a safer score. |

### Operational Conclusion
In a heavily regulated financial system, **predictive optimization is bound by the constraints of explainability**. While a Gradient Boosting model might extract an extra 2-3% in Gini coefficient (significantly reducing marginal loan losses), the structural risk of model rejection by regulators or the legal inability to explain adverse actions often forces financial institutions to stick to standard Logistic Regression + WoE architectures, or use Gradient Boosting exclusively as a challenger benchmark to discover new feature interactions.
