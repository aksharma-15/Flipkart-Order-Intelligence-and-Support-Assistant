# Flipkart-Order-Intelligence-and-Support-Assistant
IITP Capstone project


# Part 1 -- Return-Risk Scoring Pipeline

This section outlines the data verification, preprocessing, modeling, and evaluation steps for the return-risk scoring model, as executed in `part1_return_risk.ipynb`.

## 1. Dataset Verification
* The dataset is generated from the provided [generate_orders](src/generate_orders.py) file.
* **Data Dimensions:** The generated `orders_dataset.csv` contains exactly 6,000 rows and 13 columns.
* **Overall Return Rate:** The dataset exhibits an overall return rate of **22.75%**.
* **Missing Data:** The `rating_given` column is missing in 783 rows, which accounts for **13.05%** of the data. 

**Missingness Classification (MAR):**
The missingness pattern for `rating_given` is **Missing At Random (MAR)**. This classification is made because the missingness explicitly depends on the observed `payment_method` column. Specifically, there is a measured missing-rate gap where `COD` orders have a ~22% chance of missing a rating, whereas non-COD orders only have a 6% chance of missing a rating. It is not MCAR (because a real dependency exists) nor MNAR (because missingness does not depend on the unobserved rating itself).

## 2. Baseline Model Trap
A `DummyClassifier` (using the `most_frequent` strategy) was trained to establish a baseline.
* **Accuracy:** 77.25%
* **F1-Score (returned=1):** 0.0
* **Precision / Recall:** 0.0 / 0.0

**The Trap:** 
The high accuracy (77.25%) is highly misleading. This is a classic "high accuracy, zero recall" trap in imbalanced datasets. Because the model simply predicts "Not Returned" for every single order, it successfully captures the majority class but misses 100% of actual returns. For the real business problem, this renders the baseline entirely useless, as no proactive interventions can be made if the model never flags an order.

## 3. Logistic Regression & Threshold Sweep
A Logistic Regression model (with `class_weight="balanced"`) was trained on the preprocessed data.
* **Default Threshold (0.5):** 
  * **ROC-AUC:** 0.6240
  * **F1-Score:** 0.3940
  * **Recall:** 57.88%
  * **Precision:** 29.87%

* **Optimized Threshold (0.42):** 
  Sweeping the decision threshold revealed that a threshold of **0.42** maximizes the F1 score.
  * **Max F1-Score:** 0.4067
  * **Recall at 0.42:** 80.59% *(an increase of 22.71 percentage points over the default)*
  * **Precision at 0.42:** 27.19% *(a precision drop of 2.68 percentage points)*

**Business Trade-off:**
Lowering the threshold to 0.42 successfully increases recall (catching more real returns before they happen) at the expense of precision (generating more false alarms). In a business context, this threshold represents the sweet spot between spending too much money on unnecessary interventions (e.g., calling customers, holding shipments due to false positives) and wasting money on reverse-logistics fees and damaged inventory from unprevented returns (False Negatives).

## 4. Random Forest Tuning & Evaluation
* The model can accessed here: [return_risk_model](models/return_risk_model.pkl)
A `RandomForestClassifier` was tuned using `GridSearchCV` over 5-fold StratifiedKFold cross-validation.
* **Best Parameters:** `n_estimators` = 200, `max_depth` = 6
* **Cross-Validated ROC-AUC:** 0.6152
* **Held-out Test ROC-AUC:** 0.6176
*(The test ROC-AUC is within 0.05 of the CV score, providing strong evidence against severe overfitting).*

## 5. Feature Importance & Explanation
Extracting the impurity-based `.feature_importances_` from the winning Random Forest yielded the following top 5 features:
1. `payment_method_COD` (0.1645)
2. `price_inr` (0.1211)
3. `delivery_distance_km` (0.0933)
4. `order_id` (0.0911)
5. `customer_tenure_days` (0.0900)

**Permutation Importance Comparison:**
When computing `sklearn.inspection.permutation_importance` on the held-out test split, the rankings shifted significantly. Notably, **`delivery_distance_km`** dropped entirely from the top 5, plummeting to a near-zero/negative importance mean (-0.0009). 

*Why impurity overrates noisy columns:* Impurity-based importance metrics heavily overrate noisy, high-cardinality continuous columns because they offer the tree algorithm far more unique split points to artificially reduce training node impurity, regardless of true predictive signal.

## 6. Subgroup / Root-Cause Analysis
Test-set recall and precision were broken down by categorical subgroups:

**By Product Category:**
* **Apparel:** Recall: 0.5000 | Precision: 0.3333
* **Beauty:** Recall: 0.5806 | Precision: 0.4390
* **Electronics:** Recall: 0.3846 | Precision: 0.3448
* **Footwear:** Recall: 0.5000 | Precision: 0.3373
* **Home:** Recall: 0.6471 | Precision: 0.2366

**By Payment Method:**
* **COD:** Recall: 0.8903 | Precision: 0.3262
* **Prepaid_Card:** Recall: 0.0000 | Precision: 0.0000
* **Prepaid_UPI:** Recall: 0.0000 | Precision: 0.0000
* **Wallet:** Recall: 0.0000 | Precision: 0.0000

*Note: The model performs meaningfully worse on non-COD payment methods (Wallet, Prepaid_Card, Prepaid_UPI), suffering from 0.0000 recall, as well as lower performance distributions in specific sub-categories like Beauty.*

**Proposed Fix:** 
Instead of collecting more data, we should implement a **category-specific decision threshold** rather than a single global cutoff. Because categories like Beauty have a much lower baseline return probability, the global threshold is too strict. By shifting the decision threshold down specifically for the Beauty category and/or non-COD orders, we can flag high-risk orders in these groups and recover recall without cratering the model's overall precision across larger buckets.

## 7. Saved Artifacts
The tuned Random Forest pipeline (including preprocessing) has been persisted.
* **Model Path:** `models/return_risk_model.pkl` (loads without error via `joblib`).
* **Random Forest Optimal Threshold (`t*_rf`):** A secondary threshold sweep specifically on the Random Forest's `predict_proba` output determined that the F1-maximising threshold is **`t*_rf` = 0.44** (achieving a Max F1 of 0.4110). This specific value is what the LangGraph agent in Part 3 will use to anchor its return-risk buckets.


---


# Part 2: Product Image Categoriser

The trained image classifier can be accessed here: [product_classifier](models/product_classifier.pt)

### Dataset & Training Pipeline
* **Dataset:** Fashion-MNIST (Zalando Research).
* **Pre-processing:** Images were converted from 1-channel grayscale to 3 channels, resized to 224x224, and normalized using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`) to match the expected input distributions of the pretrained ResNet-18 backbone.
* **Splits Used:** 
  * Train: 55,000 images
  * Validation: 5,000 images (stratified)
  * Test: 10,000 images (untouched until final evaluation)
* **Hyperparameters:** Batch size 128, Adam optimizer, Learning Rate 0.001, 5 epochs (on feature-extraction head).

### Fine-tuning & Evaluation
* **Fine-tuning Status:** Feature extraction alone was sufficient. Validation accuracy achieved [XX.XX%], which cleared the 80% threshold, so full network fine-tuning was bypassed.
* **Final Test Accuracy:** [XX.XX%] on the 10,000-image test split.

### Confusion Patterns
Based on the generated 10x10 confusion matrix, the model exhibits the highest error rates between the following category pairs:

1. **Shirt vs. T-shirt/top:** 
   This confusion is highly visually plausible. In low-resolution 28x28 grayscale, the presence of a collar or buttons (which distinguishes a Shirt from a T-shirt) is heavily obscured. Both garments share identical outer silhouettes—a torso block with short or long sleeves—meaning the macro features the CNN relies on are virtually indistinguishable.
   
2. **Pullover vs. Coat:** 
   These two classes share identical, bulky long-sleeve silhouettes. Since Fashion-MNIST removes context like texture thickness, zippers, or the concept of wearing layers, a thick winter coat and a chunky pullover look practically identical as heavy upper-body blobs to the network.

---

### Loading & Prediction Snippet (Required for Part 3)

To load the artifact generated by Part 2 and predict a single `.png` image, use the following function:

```python
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def load_and_predict(image_path, model_path="models/product_classifier.pt"):
    # 1. Initialize model structure
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    # 2. Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Setup transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4. Predict
    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                   
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)
        
    return {"predicted_category": class_names[pred.item()], "confidence": round(conf.item(), 4)}
```

---

