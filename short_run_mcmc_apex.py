import os
import random
import argparse
import math
from time import time

import torch, torch.nn as nn
import torchvision as tv, torchvision.transforms as tr

import apex.parallel
from apex import amp


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')

    # RTX 2060 super
    # without apex:    160.69s
    # with apex (O2):   72.49s
    parser.add_argument('--apex_enabled', default=True, type=bool, help='enable apex')
    parser.add_argument('--apex_opt_level', default='O2', type=str, help='apex optimization level (O0, O1, O2, O3)')

    return parser.parse_args()


class F(nn.Module):
    def __init__(self, nc=3, nez=1, n_f=32):
        super(F, self).__init__()
        self.conv0 = nn.Conv2d(nc, n_f, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(n_f, n_f*2, 4, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(n_f*2, n_f*4, 4, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(n_f*4, n_f*8, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(n_f*8, nez, 4, 1, 0, bias=True)

    def forward(self, input, h=torch.nn.functional.leaky_relu):
        oE_l0 = h(self.conv0(input))
        oE_l1 = h(self.conv1(oE_l0))
        oE_l2 = h(self.conv2(oE_l1))
        oE_l3 = h(self.conv3(oE_l2))
        oE_out = self.conv4(oE_l3)

        return oE_out.squeeze()


def train(args):

    set_gpu(args.gpu)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    im_sz = 32
    n_ch = 3
    n_chains = 10**2
    n_f = 64
    n_i = 10**5
    K = 100

    f = F(n_f=n_f).to(device)

    transform = tr.Compose([tr.Resize(im_sz), tr.ToTensor(), tr.Normalize((.5, .5, .5), (.5, .5, .5))])
    p_d_in = tv.datasets.CIFAR10(root='data/cifar10', download=True, transform=transform)
    p_d = torch.stack([x[0] for x in p_d_in]).to(device)
    convolve = lambda x: x + 5e-2 * torch.randn_like(x)
    sample_p_d = lambda: convolve(p_d[torch.LongTensor(n_chains).random_(0, p_d.shape[0])]).detach()

    sample_p_0 = lambda: torch.FloatTensor(n_chains, n_ch, im_sz, im_sz).uniform_(-1, 1).to(device)
    def sample_q(K=K, s=1e-2):
        x_k = torch.autograd.Variable(sample_p_0(), requires_grad=True)
        for k in range(K):
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + s * torch.randn_like(x_k)
        return x_k.detach()

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=sqrt(n_chains))

    optim = torch.optim.Adam(f.parameters(), lr=1e-4, betas=[.9, .999])
    if args.apex_enabled:
        f, optim = amp.initialize(f, optim, opt_level=args.apex_opt_level)

    t0 = time()

    for i in range(n_i):
        x_p_d, x_q = sample_p_d(), sample_q()
        L = f(x_p_d).mean() - f(x_q).mean()
        optim.zero_grad()
        if args.apex_enabled:
            with amp.scale_loss(-L, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            (-L).backward()
        optim.step()

        if i % 100 == 0:
            with torch.no_grad():
                t1 = time()
                print('{:>6d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} t={:>14.2f}'.format(i, f(x_p_d).mean(), f(x_q).mean(), t1-t0))
                t0 = t1

        if i % 1000 == 0:
            plot('x_q_{:>06d}.png'.format(i), x_q)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_gpu(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    train(parse_args())