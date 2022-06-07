import numpy as np
import ot
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch import optim
from vmf_utils import hyperspherical_uniform as unif, von_mises_fisher as vmf


def Wasserstein_1D(mu_n, nu_n, p = 2, power_p = True):
    wasserstein_distance = torch.abs((torch.sort(mu_n, dim = 0)[0]* 1.0 - torch.sort(nu_n,dim = 0)[0]* 1.0))
    if power_p == True: 
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0).mean()
    else:
        wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=0).mean(), 1.0 / p)
    return wasserstein_distance
    
    
def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def sliced_wasserstein_distance(first_samples, second_samples, projections = None, num_projections=1000, p=2, power_p = True, device="cuda"):
    first_samples = first_samples.to(device)
    second_samples = second_samples.to(device)
    dim = second_samples.size(1)
    if projections == None:
        projections = rand_projections(dim, num_projections).to(device)
    else:
        projections = projections
    first_projections = torch.matmul(first_samples,projections.transpose(0, 1))
    second_projections = torch.matmul(second_samples, projections.transpose(0, 1))
    wasserstein_distance = Wasserstein_1D(first_projections, second_projections, p = p, power_p = power_p)
    return wasserstein_distance

def max_sliced_wasserstein_distance(first_samples, second_samples, lr = 0.01, p=2, max_iter=100, power_p = True, device="cuda"):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(max_iter):
        encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
        distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
        wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)
        l = -wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    
    return wasserstein_distance,theta


def distributional_sliced_wasserstein_distance(first_samples, second_samples, num_projections, f, f_op, p=2, power_p=True, max_iter=10, lam=1, device="cpu", method="NN", mu_0=None, kappa_0=None, approx_vmf=False):
    first_samples = first_samples.to(device)
    second_samples = second_samples.to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    embedding_dim = first_samples.size(1)
    if method == "NN":
        pro = rand_projections(embedding_dim, num_projections).to(device)
        f = f.to(device)
    elif method == "vMF":
        mu_t = mu_0
        # mu_t.requires_grad = True
        kappa_t = kappa_0
        kappa_t.requires_grad = True
        f_op = optim.Adam([kappa_t], lr=0.001)
    for i in range(max_iter):
        if method == "NN":
            projections = f(pro)
        elif method == "vMF":
            vmf_t = vmf.VonMisesFisher(loc=mu_t/torch.norm(mu_t), scale=kappa_t, approx=approx_vmf)
            projections = vmf_t.rsample(shape=num_projections).to(device)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p=p, power_p=power_p)
        loss = reg - wasserstein_distance
        if i % 100 == 0:
            print("Iteration {}".format(i))
            print("\t SW:{}".format(wasserstein_distance))
            print("\t Regularization:{}".format(reg))
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    
    if method == "NN":
        projections = f(pro)
    elif method == "vMF":
        vmf_t = vmf.VonMisesFisher(loc=mu_t/torch.norm(mu_t), scale=kappa_t, approx=approx_vmf)
        projections = vmf_t.rsample(shape=num_projections).to(device)

    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)
    cos = cosine_distance_torch(projections, projections)
    reg = lam * cos
    objective = wasserstein_distance - reg

    if method == "NN":
        return wasserstein_distance, projections
    elif method == "vMF":
        final_mu = (mu_t/torch.norm(mu_t)).detach()
        final_kappa = kappa_t.detach()
        return wasserstein_distance, objective, (final_mu, final_kappa)
    return wasserstein_distance, projections


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

def cosine_sum_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps))

def cost_matrix(encoded_smaples, distribution_samples, p=2):
    n = encoded_smaples.size(0)
    m = distribution_samples.size(0)
    d = encoded_smaples.size(1)
    x = encoded_smaples.unsqueeze(1).expand(n, m, d)
    y = distribution_samples.unsqueeze(0).expand(n, m, d)
    C = torch.pow(torch.abs(x - y), p).sum(2)
    return C


def cost_matrix_slow(x, y):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def compute_true_Wasserstein(X, Y, p=2):
    M = ot.dist(X, Y)
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)


def save_dmodel(model, optimizer, dis, disoptimizer, tnet, optnet, epoch, folder):
    dictionary = {}
    dictionary["epoch"] = epoch
    dictionary["model"] = model.state_dict()
    dictionary["optimizer"] = optimizer.state_dict()
    if not (disoptimizer is None):
        dictionary["dis"] = dis.state_dict()
        dictionary["disoptimizer"] = disoptimizer.state_dict()
    else:
        dictionary["dis"] = None
        dictionary["disoptimizer"] = None
    if not (tnet is None):
        dictionary["tnet"] = tnet.state_dict()
        dictionary["optnet"] = optnet.state_dict()
    else:
        dictionary["tnet"] = None
        dictionary["optnet"] = None

    torch.save(dictionary, folder + "/model.pth")


def load_dmodel(folder):
    dictionary = torch.load(folder + "/model.pth")
    return (
        dictionary["epoch"],
        dictionary["model"],
        dictionary["optimizer"],
        dictionary["tnet"],
        dictionary["optnet"],
        dictionary["dis"],
        dictionary["disoptimizer"],
    )


def compute_Wasserstein(x, y, device, p=2):
    M = cost_matrix(x, y, p)
    pi = ot.emd([], [], M.cpu().detach().numpy())
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi * M)


def make_spiral(n_samples, noise=.5):
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.array(np.hstack((d1x,d1y)))

get_rot= lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def get_data(n_samples,theta,scale=1,transla=0):
    Xs = make_spiral(n_samples=n_samples, noise=1)-transla
    Xt = make_spiral(n_samples=n_samples, noise=1)
    
    A=get_rot(theta)
    
    Xt = (np.dot(Xt,A))*scale+transla
    
    return Xs,Xt



def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def spiral(N=100,phi=0):

    theta = np.sqrt(np.random.rand(N))*4*pi # np.linspace(0,2*pi,100)    
    r_a = theta/2 + pi
    data_a = np.array([np.cos(theta + phi)*r_a, np.sin(theta+phi)*r_a]).T
    x_a = data_a + np.random.randn(N,2)*0.3
    
    return torch.from_numpy(x_a).float()

    
def prior_similarity_vector(maxi,dim,nb_samples,tau, device):
    prior_samples = rand_projections(dim, nb_samples*10).to(device)
    inner_prod = prior_samples@maxi.detach().T
    ind = torch.where(torch.abs(inner_prod)> 1 - tau)[0]
    return  prior_samples[ind]
        
        
class TransformNet(nn.Module):
    """
    used usually for changing the distribution of the random projection
    """
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size,self.size),
                                 nn.LeakyReLU(),
                                 nn.Linear(self.size,self.size),
                                 nn.LeakyReLU(),
                                 nn.Linear(self.size,self.size))
    def forward(self, input):
        out =self.net(input)
        return out/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))


class TransformLatenttoOrig(nn.Module):
    """
    used for mapping the random projection vector into the the ambient space
    of the distribution
    """
    def __init__(self, dim_latent,dim_orig,dim_hidden=10):
        super(TransformLatenttoOrig, self).__init__()
        self.dim_latent = dim_latent
        self.dim_orig = dim_orig
        self.dim_hidden = dim_hidden
        self.net = nn.Sequential(nn.Linear(self.dim_latent,self.dim_hidden),
                                 #nn.Sigmoid(),
                                 nn.ReLU(),
                                 nn.Linear(self.dim_hidden,self.dim_hidden),
                                 #nn.Sigmoid(),
                                 nn.ReLU(),
                                nn.Linear(self.dim_hidden,self.dim_hidden),
                                 #nn.Sigmoid(),
                                 nn.ReLU(),
                                 nn.Linear(self.dim_hidden, self.dim_orig),
                                 nn.Sigmoid(),
                                 #nn.ReLU(),
                                 )
    def forward(self, input):
        out =self.net(input)
        return out/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))


def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix_slow(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix_slow(x2, theta * r))
    wasserstein_distance = torch.abs(
        (torch.sort(cost_matrix_1.transpose(0, 1), dim=1)[0] - torch.sort(cost_matrix_2.transpose(0, 1), dim=1)[0])
    )
    wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance


def rescale(X):
    maxx = torch.max(X, dim = 1)[0].reshape(-1,1)
    minn = torch.min(X, dim = 1)[0].reshape(-1,1)
    return (X - minn) / (maxx - minn)

def generateFMNIST(class_1 = 4, class_2 = 5, scale = True, device = "cuda"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,),)])
    training_set = datasets.FashionMNIST(root="data",train=True,download=True,transform=transform) 
    test_set = datasets.FashionMNIST(root="data",train=False,download=True,transform=transform)
    
    class_1 = class_1
    class_2 = class_2
    
    Xt_train = training_set.data[(training_set.targets==class_1)]
    Xt_train = Xt_train.reshape(Xt_train.shape[0], Xt_train.shape[1]*Xt_train.shape[2])
    Xs_train = training_set.data[(training_set.targets==class_2)]
    Xs_train = Xs_train.reshape(Xs_train.shape[0], Xs_train.shape[1]*Xs_train.shape[2])
    
    Xt_test= test_set.data[(test_set.targets==class_1)]
    Xt_test = Xt_test.reshape(Xt_test.shape[0], Xt_test.shape[1]*Xt_test.shape[2])
    Xs_test = test_set.data[(test_set.targets==class_2)]
    Xs_test = Xs_test.reshape(Xs_test.shape[0], Xs_test.shape[1]*Xs_test.shape[2])
    
    if scale is True:
        Xt_train = rescale(Xt_train.float().to(device))
        Xs_train = rescale(Xs_train.float().to(device))
        Xt_test = rescale(Xt_test.float().to(device))
        Xs_test = rescale(Xs_test.float().to(device))

    return Xt_train, Xs_train, Xt_test, Xs_test


