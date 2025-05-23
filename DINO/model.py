import os
import cv2
import torch
from torch import nn
# import torchvision.models as models
# import timm
# from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import math

import vision_transformer as vits




class model:
    def __init__(self):
        self.checkpoint = "best_model.pth"
        self.dino = 'checkpoint.pth'
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")
        self.transform = transforms.Compose([
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # self.feat_extractor = vits.__dict__['vit_small'](patch_size=16)

        # state_dict = torch.load('checkpoint.pth')

        # state_dict = state_dict['teacher']

        # # remove `module.` prefix
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # # remove `backbone.` prefix induced by multicrop wrapper
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # # pretrained_model = model.load_state_dict(state_dict, strict=False)
        # self.feat_extractor.load_state_dict(state_dict, strict=False)
        self.model = ABMIL(dim_in=384, dim_out=5)


    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """

        # join paths

        self.feat_extractor = vits.__dict__['vit_small'](patch_size=16)

        state_dict = torch.load(os.path.join(dir_path, self.dino), map_location=self.device, weights_only=False)

        state_dict = state_dict['teacher']

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # pretrained_model = model.load_state_dict(state_dict, strict=False)
        self.feat_extractor.load_state_dict(state_dict, strict=False)

        checkpoint_path = os.path.join(dir_path, self.checkpoint)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        new_ckpt = {}
        for name, param in ckpt.items():
            new_name = name.replace('_orig_mod.module.', '')
            new_ckpt[new_name] = param 
            # print(name)
        # model.load_state_dict(torch.load(checkpoint_path))
        self.model.load_state_dict(new_ckpt)

        # self.model.load_state_dict(torch.load(
        #     checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    # def predict(self, input_image, patient_info_dict):
    def predict(self, input_image):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: an int value indicating the class for the input image.
        """
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Convert the image from numpy array to PIL Image
        img_pil = Image.fromarray((img).astype(np.uint8))

        patch_size = 200
        stride = 100

        # Calculate the number of patches in each dimension
        num_patches_x = (img.shape[1] - patch_size) // stride + 1
        num_patches_y = (img.shape[0] - patch_size) // stride + 1

        idx = 0 

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                idx += 1
                start_x = j * stride
                start_y = i * stride
                end_x = start_x + patch_size
                end_y = start_y + patch_size

                patch = img_pil.crop((start_x, start_y, end_x, end_y))

                patch_tensor = self.transform(patch)
                patch_tensor = patch_tensor.unsqueeze(dim=0)
                feature_tensor = self.feat_extractor(patch_tensor)

                if idx == 1:
                    feature_box = feature_tensor
                    continue
                else:
                    feature_box = torch.cat((feature_box, feature_tensor), dim=0)
        
        feature_box = feature_box.unsqueeze(dim=0)
        with torch.no_grad():
            score, attention = self.model(feature_box)
        _, pred_class = torch.max(score, 1)
        pred_class = pred_class.detach().cpu()

        return int(pred_class), attention
    
        # _img = Image.open(input_image).convert('RGB')
        # img = self.transform(_img)
        # image = img.to(self.device, torch.float)
        # image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image_rgb, (800, 800))
        # image = image / 255

        # image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        # image = image.to(self.device, torch.float)





# def split_image_into_patches_and_generate_features(input_image, patch_size, stride):
#     # Find image name
#     # img_name = img_path.split('/')[-1].split('.')[0]

#     # Load image
#     img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

#     # Convert the image from numpy array to PIL Image
#     img_pil = Image.fromarray((img).astype(np.uint8))

#     # Calculate the number of patches in each dimension
#     num_patches_x = (img.shape[1] - patch_size[0]) // stride + 1
#     num_patches_y = (img.shape[0] - patch_size[1]) // stride + 1

#     # os.makedirs(save_path, exist_ok=True)
#     idx = 0

#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             # Define the coordinates of the top-left and bottom-right corners of the patch
#             idx += 1
#             start_x = j * stride
#             start_y = i * stride
#             end_x = start_x + patch_size[0]
#             end_y = start_y + patch_size[1]a

#             # Extract the patch from the image
#             patch = img_pil.crop((start_x, start_y, end_x, end_y))

#             patch_tensor = feature_encoder(patch)

#             if idx == 1:
#                 feature_box = patch_tensor
#                 continue
#             else:
#                 feature_box = torch.cat((feature_box, patch_tensor), dim=0)
    
#     return feature_box
            
            # # Save the patch as a separate image file
            # patch_file = os.path.join(save_path, f'{img_name}_{i}_{j}.png')
            # patch.save(patch_file)

class ABMIL(nn.Module):
    def __init__(self, dim_in, L=512, D=128, K=1, dim_out=2, dropout=0.):
        super(ABMIL, self).__init__()
        self.L, self.D, self.K = L, D, K

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, self.L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.L, dim_out)

    def bag_forward(self, bag):
        H = self.encoder(bag)  # NxL

        A = self.attention(H)  # Nx1
        A = torch.transpose(A, 1, 0)  # 1xN
        A = F.softmax(A, dim=1)  # softmax over N
        A = A / math.sqrt(A.shape[-1])
        # print(H.shape)
        # print(A.shape)
        M = torch.mm(A, H)  # 1xL

        output = self.decoder(M)
        output = self.fc(output)
        return output, A

    def batch_forward(self, batch):
        outputs = []
        for bag in batch:
            outputs.append(self.bag_forward(bag))
        return torch.cat(outputs, 0)

    def forward(self, x):  # B x N x dim_in, a bag
        if isinstance(x, list):
            outputs = self.batch_forward(x)
        elif isinstance(x, torch.Tensor):
            if x.shape[0] == 1:
                outputs = self.bag_forward(x.squeeze(0))
            else:
                outputs = self.batch_forward(x)
        else:
            raise TypeError
        return outputs


# if __name__ == '__main__':
#     path = '/data_sdb/THUHITCoop/1. Classification of Myopic Maculopathy/category/3/mmac_task_1_train_1126.png'
#     input_image = cv2.imread(path, 1)
    
#     img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

#     model = model()
#     # model.predict(input_image)

#     # Convert the image from numpy array to PIL Image
#     img_pil = Image.fromarray((img).astype(np.uint8))

#     patch_size = 200
#     stride = 200

#     # Calculate the number of patches in each dimension
#     num_patches_x = (img.shape[1] - patch_size) // stride + 1
#     num_patches_y = (img.shape[0] - patch_size) // stride + 1

#     idx = 0 

#     device = torch.device("cpu")

#     feat_extractor = vits.__dict__['vit_small'](patch_size=16)

#     state_dict = torch.load('/data_sdb/THUHITCoop/ljw/TEST/checkpoint.pth', map_location=device)

#     state_dict = state_dict['teacher']

#     # remove `module.` prefix
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     # remove `backbone.` prefix induced by multicrop wrapper
#     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

#     # pretrained_model = model.load_state_dict(state_dict, strict=False)
#     feat_extractor.load_state_dict(state_dict, strict=False)

#     transform = transforms.Compose([
#             # transforms.Resize([224, 224]),
#             transforms.ToTensor(),
#             # transforms.RandomResizedCrop(224),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])

#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             idx += 1
#             start_x = j * stride
#             start_y = i * stride
#             end_x = start_x + patch_size
#             end_y = start_y + patch_size

#             patch = img_pil.crop((start_x, start_y, end_x, end_y))

#             patch_tensor = transform(patch)
#             patch_tensor = patch_tensor.unsqueeze(dim=0)
#             feature_tensor = feat_extractor(patch_tensor)

#             if idx == 1:
#                 feature_box = feature_tensor
#                 continue
#             else:
#                 feature_box = torch.cat((feature_box, feature_tensor), dim=0)
    
#     milnet = ABMIL(dim_in=384, dim_out=5)
#     feature_box = feature_box.unsqueeze(dim=0)
#     with torch.no_grad():
#         score = milnet(feature_box)
#     _, pred_class = torch.max(score, 1)
#     pred_class = pred_class.detach().cpu()

#     # model = model(checkpoint, dino, device, transform)
#     # model.predict(input_image=input_image)
#     # print('ok')