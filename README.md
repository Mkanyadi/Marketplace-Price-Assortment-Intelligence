"""# Price Prediction + Opportunity Detection (ML End-to-End)

## Problema
Proiectul estimeaza pretul de inchiriere pe zi (regresie) si identifica oportunitati (clasificare) pentru nise cu oferta mica si pret potential mare.

## Dataset
- Sursa: https://huggingface.co/datasets/flodussart/get_around_pricing_optimization
- Dimensiune: ~4.8k randuri, 15 coloane
- Ce contine: date despre anunturi de inchiriere auto (caracteristici precum model_key, car_type, mileage, engine_power, fuel etc.) si pretul pe zi (rental_price_per_day).
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

## Nota despre modele (important)
- Modelul de regresie pentru pret (price_model.pkl) NU este inclus in repository deoarece depaseste limita de upload din browser (25MB).
- Modelul se poate recrea prin rularea notebook-ului (celulele de training), apoi se salveaza local in folderul `models/`.
- Modelul de clasificare (opportunity_model.pkl) este inclus in repo.
"""
