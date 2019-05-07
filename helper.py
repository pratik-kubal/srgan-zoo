import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tensorlayer.prepro import *

class DataLoader():
    def __init__(self, dataset_name, img_res,batch_size,path):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.batch_size = batch_size
        self.path = glob(path+'/%s/*' % (self.dataset_name))

    def load_data(self, is_testing=False):

        batch_images = np.random.choice(self.path, size=self.batch_size)
        imgs_hr = []
        imgs_lr = []
    
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = crop(img,wrg=w, hrg=h,is_random=True)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    def imread(self, path):
        return imageio.imread(path).astype(np.float)
    
def sample_images(gen, epoch, dataset_name,imgs_hr, imgs_lr):
        os.makedirs('images/%s' % dataset_name, exist_ok=True)
        r, c = 2, 2

        fake_hr = gen.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' %
                        (dataset_name, epoch, i))
            plt.close()
