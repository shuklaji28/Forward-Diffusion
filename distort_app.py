import streamlit as st
import cv2
import torch
from pprint import pprint
import PIL
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import urllib

st.set_page_config(layout='wide')
device = '0' if torch.cuda.is_available() else 'cpu'

#let's also make a reverse transform function which will take us back from Torch tensor to PIL
#Compose function combines multiple transformations.
#in general we have image dimensions as (height, width, channels), so we'll apply permute function here as transformation to swap the dimensions. Usually Torch has channels as second dimension. Here we'll change and make it last also without introducing any batch size like we earlier mentioned initially. We are making it general here in this transformation.
# In the context of image processing, it's common to have tensors in the format (channels, height, width), and permute can be used to rearrange the dimensions accordingly.

IMAGE_SHAPE = (32,32)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SHAPE),
    transforms.ToTensor(), #this function transforms/scales pixel values in the range from [0 - 1]
    transforms.Lambda(lambda t: (t*2)-1) #this maps pixel values within the range [-1 to 1]. 0 -> -1 and 1 -> 1
    ])

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1,2,0)),
    transforms.Lambda(lambda t: t*255.),
    transforms.Lambda(lambda t : t.numpy().astype(np.uint8)),
    transforms.ToPILImage()

])

def get_sample_image() -> PIL.Image.Image:
    url = 'http://tse4.mm.bing.net/th/id/OIP.dnsx9J5LmZoiMT3sQ2MJEQHaE8?w=273&h=182&c=7&r=0&o=5&dpr=1.5&pid=1.7'
    filename = 'water_meme.jpg'
    urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)

class DiffusionModel():
    
    def __init__(self, start_schedule = 0.0001, end_schedule = 0.02, timesteps = 300):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps
        
        # '''
        # here if betas = [0.1, 0.2, 0.3, ....]
        # then we have our alphas as 1-betas, which is
        #  alphas  = [0.9, 0.8, 0.7 ....]
        #  alphas_cumprod  = [0.9,   0.9*0.8,  0.9*0.8*0.7, ....]
         
        #  here we'll be assigning these variables below.
        #  '''
        
        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        
    def forward(self, x_0, t, device):
        
        # '''
        # remember, our image tensor has shape : (B, C, H, W)
        # and t: (B,)
        # '''
        
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t , x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t , x_0.shape)
        
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance  = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        
        return mean+variance, noise.to(device)
    
    
    def get_index_from_list(self, values, t, x_shape):
        batch_size = t.shape[0]
        
        result = values.gather(-1, t.cpu())
        
        
        # '''
        # what we are doing here is that if 
        # x_shape = (5,3, 64,64)
        # then len(x_shape) = 4
        # and len(x_shape)-1 = 3
        
        # thus we reshape 'out' to dimension - (batch_size, 1,1 1)
        
        # '''
        
        
        return result.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)


def forward_diff(img, timestamp):
    # pil_image = get_sample_image()
    pil_image = Image.open(img)
    torch_image = transform(pil_image)

    diffusion_model = DiffusionModel()

    NO_DISPLAY_IMAGES = timestamp

    torch_image_batch = torch.stack([torch_image]*NO_DISPLAY_IMAGES)
    t = torch.linspace(0, diffusion_model.timesteps -1, NO_DISPLAY_IMAGES).long()
    noisy_image_batch, noise_list = diffusion_model.forward(torch_image_batch, t, device)
    return noisy_image_batch, noise_list
    

def main():
    st.title("Visualize Forward Diffusion Process")
    uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_image:
        timestamp = st.slider("Timestamp", min_value=1, max_value=50, value=1)
        st.write(f"Timestamp selected: {timestamp}")
        images = forward_diff(uploaded_image, timestamp)[0]
        # cols = st.columns(10)
        # for i, img in enumerate(images):
        #     with cols[i]:
        #         st.image(reverse_transform(img), caption='Forward passed Image', use_column_width=True)
                # Calculate the number of rows needed
        num_rows = len(images) // 10 + (len(images) % 10 > 0)

        # Create a grid layout
        cols = st.columns(10)

        # Iterate over rows
        for row in range(num_rows):
            # Iterate over columns
            for col in range(10):
                img_index = row * 10 + col
                if img_index < len(images):
                    with cols[col]:
                        st.image(reverse_transform(images[img_index]), use_column_width=True)
    
    
    # st.header("Visualize Noise")



    # st.balloons()


if __name__ == '__main__':
    
    main()