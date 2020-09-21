import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
model.eval()

f_path = 'me.jpg' # replace 'me.jpg' with your img.
input_image = Image.open(f_path)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
	input_batch = input_batch.to('cuda')
	model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

if torch.cuda.is_available(): #output_prediction is equal to the index of the class we want to keep in the resulting image
	person_mask = torch.where(output_predictions ==15, torch.tensor(1).to('cuda'),torch.tensor(0).to('cuda'))
else:	
	person_mask = torch.where(output_predictions ==15, torch.tensor(1),torch.tensor(0))
mask = np.zeros_like(input_image)
for i in range(3):
    mask[:,:,i]=person_mask.cpu().numpy()

removed_bg = mask*input_image

# plt.imshow(removed_bg) #uncomment to show image
# plt.show()   
plt.imsave('removed_bg.jpg',removed_bg)
