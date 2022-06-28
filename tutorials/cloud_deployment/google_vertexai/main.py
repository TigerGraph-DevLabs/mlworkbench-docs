
import torch
import kserve
from google.cloud import storage
# from sklearn.externals import joblib
from kserve import Model, Storage
from kserve.model import ModelMissingError, InferenceError
from typing import Dict
import logging
import pyTigerGraph as tg
import os 
import sys
import json

logger = logging.getLogger(__name__)

class VertexClassifier(Model):
    def __init__(self, name: str, source_directory: str):
        super().__init__(name)
        self.name = name
        self.source_dir = source_directory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Load configuration JSON file
        with open(os.path.join(source_directory, "config.json")) as json_file:
            data = json.load(json_file)
            self.model_config = data["model_config"]
            connection_config = data["connection_config"]
            loader_config = data["infer_loader_config"]
            model_name = data["model_name"]

        # Setup Connection to TigerGraph Database
        self.conn = tg.TigerGraphConnection(**connection_config)

        # Setup Inference Loader
        self.infer_loader = self.conn.gds.neighborLoader(**loader_config)

        # Setup Model
        self.model_name = model_name
        self.model = self.load_model(model_name)
        

    def load(self):
        pass
    
    def load_model(self, name):
        import gat_cora.model as model
        mdl = getattr(model, name)(**self.model_config)
        logger.info("Instantiated Model")
        with open(os.path.join(self.source_dir, "model.pth"), 'rb') as f:
            mdl.load_state_dict(torch.load(f))
        mdl.to(self.device).eval()
        logger.info("Loaded Model")
        return mdl

    def predict(self, request: Dict) -> Dict:
        input_nodes = request["vertices"]
        input_ids = set([str(node['primary_id']) for node in input_nodes])
        logger.info(input_ids)
        data = self.infer_loader.fetch(input_nodes).to(self.device)
        logger.info (f"predicting {data}")
        with torch.no_grad():
            output = self.model(data)
        returnJSON = {}
        for i in range(len(input_nodes)):
            returnJSON[input_nodes[i]["primary_id"]] = list(output[i].tolist())
        return returnJSON

if __name__ == "__main__":
    model_name = os.environ.get('K_SERVICE', "tg-gat-gcp-demo-predictor-default")
    model_name = '-'.join(model_name.split('-')[:-2]) # removing suffix "-predictor-default"
    logging.info(f"Starting model '{model_name}'")
    model = VertexClassifier(model_name, "./gat_cora")
    kserve.ModelServer(http_port=8080).start([model])
