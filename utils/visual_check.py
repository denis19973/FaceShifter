import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torchvision
import PIL.Image as Image




class VisualChecker:
    """
    Class for saving intermediate generator results for two specific photos.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    save_path = './gen_images/'
    Xs_numpy_arr = None
    Xt_numpy_arr = None
    Xs_tensor = None
    Xt_tensor = None
    source_embedding = None
    target_embedding = None

    def __init__(self,  source_path: str, target_path: str, embedder, save_path: str = None):
        self.Xs_numpy_arr = np.array(Image.fromarray(cv2.imread(source_path)))
        self.Xt_numpy_arr = np.array(Image.fromarray(cv2.imread(target_path)))

        Xs_transformed = self.apply_transform(self.Xs_numpy_arr)
        Xt_transformed = self.apply_transform(self.Xt_numpy_arr)

        self.Xs_tensor = Xs_transformed.unsqueeze(0).cuda()
        self.Xt_tensor = Xt_transformed.unsqueeze(0).cuda()

        self.source_embedding, _ = embedder(F.interpolate(self.Xs_tensor[:, :, 19:237, 19:237], (112, 112),
                                          mode='bilinear', align_corners=True))
        self.target_embedding, __ = embedder(F.interpolate(self.Xt_tensor[:, :, 19:237, 19:237], (112, 112),
                                           mode='bilinear', align_corners=True))

        if save_path:
            self.save_path = save_path

    def apply_transform(self, tens):
        return self.transform(tens)

    def save_visual_result(self, generator, epoch):
        with torch.no_grad():
            Yt, _ = generator(self.Xt_tensor, self.source_embedding)
            Ys, _ = generator(self.Xs_tensor, self.target_embedding)

            image = self.make_image(self.Xs_tensor, self.Xt_tensor, Ys, Yt)
            print('image.channels', image.shape, image.transpose([1, 2, 0]).shape)
            cv2.imwrite(f'{self.save_path}visual_check_{epoch}.jpg', image)

    def get_grid_image(self, X):
        return X.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5

    def make_image(self, Xs, Xt, Ys, Yt):
        Ys = self.get_grid_image(Ys)
        Yt = self.get_grid_image(Yt)
        Y = np.concatenate((Ys, Yt), axis=1)
        X = np.concatenate((self.Xs_numpy_arr / 255., self.Xt_numpy_arr / 255.), axis=1)
        print('shapes', X.shape, Y.shape)
        return np.concatenate((X, Y), axis=0)
