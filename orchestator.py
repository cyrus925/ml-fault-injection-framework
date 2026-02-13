from ingestion.bronze_to_silver import Bronze
from ingestion.silver_to_gold import Silver
from ml.create_features import Features
from ml.fit_model import Fit
from ml.predict import Predict
from ingestion.inject_players_faults import FaultPlayerInjector
from ml.predict import Predict



# Dataset / pipeline target
name="players"

#Transform Bronze -> Silver
bronze = Bronze(name)
if  not bronze.target_path.exists():
    print(f'Silver table is being written -> {bronze.target_path}')
    bronze.ingest()

#Transform Silver -> Gold
silver= Silver(name)
if not silver.target_path.exists():
    print(f'Gold table is being written -> {silver.target_path}')
    silver.load() 

#Create ML features from the Gold
create_features = Features(name)
if not create_features.features_path.exists():
    print(f'Features table is being written -> {create_features.features_path}')
    create_features.build_features()

#Train ML model
fit_model = Fit(name)
if not fit_model.model_path.exists():
    print(f'Model table is being written -> {fit_model.model_path}')
    fit_model.train()



#Simulate missing or corrupted data
inject_fault= FaultPlayerInjector()
if not inject_fault.output_path.exists():
    print(f'Inject table is being written -> {inject_fault.output_path}')
    inject_fault.run()

#Run predictions on fault-injected dataset
predict_fault=Predict(name)
if not predict_fault.output_path.exists() and inject_fault.output_path.exists():
    print(f'Pred table is being written -> {predict_fault.output_path}')
    predict_fault.predict()


 
