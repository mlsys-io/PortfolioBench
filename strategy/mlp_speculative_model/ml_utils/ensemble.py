import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "models", "model_final_5_2.h5")

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras  # type: ignore


class SimpleModel:
    def __init__(self, path):
        print("Loading model...")
        self.model = keras.models.load_model(path)
        print("Model loaded")
        
    def predict(self, data):
       numeric_data = data.select_dtypes(include=[np.number])
       X = StandardScaler().fit_transform(numeric_data.values)
       return np.argmax(self.model.predict(X), axis=1) 

sample_model = SimpleModel(file_path)

# class EnsembleLearner:
#     def __init__(self, model_paths=[
#         "./models/model_final_2_1.h5",
#         "./models/model_final_2_2.h5",
#         "./models/model_final_5_2.h5",
#         "./models/model_final_3_2.h5",
#         "./models/model_final_4_2.h5"
#     ]):
#         self.models = [SimpleModel(path) for path in model_paths]

#     def predict(self, data):
#         predictions = np.array([model.predict(data) for model in self.models])
#         predictions = predictions.T
#         voted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
#         return voted_labels
    

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # from ml_utils.data_process import DataProcess
    # test_data_path = "./ml_model/data/ETH_USDT-4h.feather"

    # data_processor = DataProcess(test_data_path)
    # test_df = data_processor.df

    # ensemble = EnsembleLearner()
    # predictions = ensemble.predict(test_df)
    # print("Ensemble predictions:", predictions)
    
    print(file_path)
    print(sample_model)

