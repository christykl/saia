import os
import numpy as np
import base64
from PIL import Image

class SyntheticExemplars():
    
    def __init__(self, path2exemplars, path2save, mode, n_exemplars=15, im_size=224):
        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.im_size = im_size
        self.mode = mode

        self.exemplars = {}
        self.scores = {}

        exemplars, scores = self.net_dissect()
        self.exemplars[mode] = exemplars
        self.scores[mode] = scores

    def net_dissect(self,im_size=224):
        print(self.path2exemplars)
        activations = np.loadtxt(os.path.join(self.path2exemplars,'activations.csv'), delimiter=',')
        image_array = np.load(os.path.join(self.path2exemplars,'images.npy'))
        all_images = []
        if len(activations.shape) == 1:
            activations = activations.reshape(1, len(activations))
        for unit in range(activations.shape[0]):
            curr_image_list = []
            for exemplar_inx in range(min(activations.shape[1],self.n_exemplars)):
                save_path = os.path.join(self.path2save,'synthetic_exemplars',str(unit))
                if os.path.exists(os.path.join(save_path,f'{exemplar_inx}.png')):
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(image)
                else:
                    curr_image = image_array[unit,exemplar_inx]
                    curr_image =  Image.fromarray(curr_image.astype(np.uint8))
                    os.makedirs(save_path,exist_ok=True)
                    curr_image.save(os.path.join(save_path,f'{exemplar_inx}.png'), format='PNG')
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        curr_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(curr_image)
            all_images.append(curr_image_list)

        return all_images,activations[:,:self.n_exemplars]