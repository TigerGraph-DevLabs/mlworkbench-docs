import torch
import pyTigerGraph as tg
import json
import model
import os

def model_fn(model_dir):
    with open(os.path.join(model_dir, "code/config.json")) as json_file:
        config = json.load(json_file)
    connection_config = config["connection_config"]
    model_config = config["model_config"]
    loader_config = config["infer_loader_config"]
    model_name = config["model_name"]

    mdl = getattr(model, model_name)

    conn = tg.TigerGraphConnection(**connection_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = mdl(**model_config)
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        gnn_model.load_state_dict(torch.load(f))
    gnn_model.to(device).eval()

    infer_loader = conn.gds.neighborLoader(**loader_config)

    model_loader_dict = {"model": gnn_model, "loader": infer_loader}

    return model_loader_dict
    
def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        input_data = json.loads(request_body)
        verts = input_data["vertices"]
        return verts
    else:
        raise Exception("Requested unsupported ContentType in content_type: {}".format(content_type))

def predict_fn(input_data, model):
    loader = model["loader"]
    gnn = model["model"]
    sub_graphs = loader.fetch(input_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    sub_graphs.to(device)
    with torch.no_grad():
        output = gnn(sub_graphs)
    return (input_data, output.cpu())

def output_fn(prediction, content_type):
    if content_type == "application/json":
        returnJson = {}
        for i in range(len(prediction[0])):
            returnJson[prediction[0][i]["primary_id"]] = list(prediction[1][i].tolist())
        return json.dumps(returnJson)
    raise Exception("Requested unsupported ContentType in content_type: {}".format(content_type))
