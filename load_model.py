import torch
import torchvision.transforms as transforms
from PIL import Image
from models import suge # replace with your actual model file

# Create model instance
model =

# Load saved weights
model.load_state_dict(torch.load("models/sugarcane_disease_model.pth"))
model.eval()  # set to evaluation mode
