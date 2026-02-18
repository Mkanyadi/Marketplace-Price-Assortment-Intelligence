# Price Prediction + Opportunity Detection (ML End-to-End)

## Problema
Proiectul estimeaza pretul de inchiriere pe zi (regresie) si identifica oportunitati (clasificare) pentru nise cu oferta mica si pret potential mare.

## Dataset
- Sursa: HuggingFace dataset (car rental)
- Target regresie: rental_price_per_day
- Label clasificare: opportunity (definit heuristic: supply mic + pret estimat mare)

## Ce am facut
1. EDA (vizualizari + observatii)
2. Preprocesare (OneHot pentru categorice, StandardScaler pentru numerice)
3. Model 1 (Regresie): Ridge vs RandomForestRegressor (final: RandomForest)
4. Model 2 (Clasificare): LogisticRegression pentru opportunity + Confusion Matrix
5. Export rezultate (CSV) si salvare modele (PKL)

## Rezultate 
- Regresie:
  - Ridge: MAE=12.117, RMSE=17.970
  - RandomForest: MAE=10.740, RMSE=16.941 (model final)
- Clasificare (Opportunity, LogisticRegression):
  - Accuracy=0.9866, Precision=0.9613, Recall=0.9551, F1=0.9582

## Cum rulezi
pip install -r requirements.txt
Ruleaza notebook-ul in Google Colab / Jupyter.

## Fisiere generate
- models/price_model.pkl
- models/opportunity_model.pkl
- results/top_opportunities.csv
- results/opportunity_summary_by_model_key.csv
- results/opportunity_summary_by_car_type.csv
