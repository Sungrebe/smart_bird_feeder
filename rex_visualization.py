from rex_xai.config import CausalArgs
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor
from rex_xai.explanation import load_and_preprocess_data, predict_target, calculate_responsibility, analyze
from rex_xai.extraction import Explanation
from rex_xai._utils import get_device, Strategy
from rex_xai.multi_explanation import MultiExplanation

#from transformers import CLIPProcessor, CLIPVisionModel

import copy
import torch
#from torchvision.models import resnet50
import timm
from torchvision import transforms as T
from PIL import Image

import torch as tt
import torch.nn.functional as F

args = CausalArgs()

args.gpu = False
args.iters = 10
args.seed = 123

#model = resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
model = timm.create_model("resnet50d.ra2_in1k", pretrained=True)
model.reset_classifier(0)
model.load_state_dict(torch.load('finetuned_model.pth'))
#model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()
model.to("cpu")

model_shape = ["N", 3, 224, 224]

def preprocess(path, shape, device, mode) -> Data:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode=mode, process=False)
    data.data = transform(img).unsqueeze(0).to(device)
    data.mode = "RGB"
    data.model_shape = shape
    data.model_height = 224
    data.model_width = 224 
    data.model_channels = 3
    data.transposed = True
    data.model_order = "first"

    return data

def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        #tensor = model.vision_model(mutants)
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor)
        #return from_pytorch_tensor(tensor.pooler_output)

device = get_device(gpu=False)

args.path = 'original.jpg'
data = load_and_preprocess_data(model_shape, device, args)

data.set_mask_value(0)
data.target = predict_target(data, prediction_function)

resp_maps, stats = calculate_responsibility(data, args, prediction_function)
exp = Explanation(resp_maps, prediction_function, data, args, stats)
exp.extract(Strategy.Global)

exp.show('explanation_jay_finetuned.jpg')

# multi_exp = MultiExplanation(resp_maps, prediction_function, data, args, stats)
# multi_exp.extract(Strategy.MultiSpotlight)
# multi_exp.show("explanation_jay_finetuned_multi.jpg")