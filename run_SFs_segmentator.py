import torch
from torch.autograd import Variable
from preprocess import get_loader
from torchvision.utils import save_image
from tqdm import tqdm
from U2_Net import U2NET
import os
import config


def main():
    params = config.parse_args()
    device = torch.device('cuda:0')

    model_dir = 'model/'
    test_dir = 'validation/' + 'example/'
    save_dir = 'results/' + 'example/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_data_loader = get_loader(image_path=test_dir,
                                  image_size=params.resolution,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=params.num_workers,
                                  mode='test',
                                  augmentation_prob=0.)

    if params.RGB is True:
        input_dim = 3
        output_dim = 3
    else:
        input_dim = 1
        output_dim = 1

    G = U2NET(in_ch=input_dim, out_ch=output_dim)
    G.to(device)
    G.load_state_dict(torch.load(model_dir + 'generator.pkl'))
    # G.eval()

    for i, (input, target) in enumerate(test_data_loader):
        # input & target image data
        with torch.no_grad():
            x_ = Variable(input.to(device))
            # G.eval()
            gen_image, _, _, _, _, _, _ = G(Variable(x_.to(device)))
            path = save_dir + "%04d.png" % i
            save_image(gen_image.data,
                       os.path.join(path), nrow=4, scale_each=True)
            print('%d images are generated.' % (i + 1))


if __name__ == '__main__':
    main()
