#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data_utils
from utils import rand_projections, TransformNet, Wasserstein_1D, sliced_wasserstein_distance
from tqdm import tqdm
from vmf_utils import hyperspherical_uniform as unif, von_mises_fisher as vmf


class Discriminator(nn.Module):
    """
    used as a discriminator for computing KL 
    """
    def __init__(self, dim,dim_hidden=10):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.net = nn.Sequential(nn.Linear(self.dim,self.dim_hidden),
                                 nn.Sigmoid(),
                                 #nn.ReLU(),
                                 nn.Linear(self.dim_hidden,self.dim_hidden),
                                 nn.Sigmoid(),
                                 #nn.ReLU(),
                                 nn.Linear(self.dim_hidden,self.dim_hidden),
                                 nn.Sigmoid(),
                                 #nn.ReLU(),
                                 nn.Linear(self.dim_hidden, 1),
                                 #nn.Sigmoid(),
                                 #nn.ReLU(),
                                 )
    def forward(self, input):
        out =self.net(input)
        return out

def  create_data_loader(X, batch_size):
    #print(X.shape)
    data = data_utils.TensorDataset(X)
    return data_utils.DataLoader(data, batch_size= batch_size, drop_last = False,sampler = data_utils.sampler.RandomSampler(data))

def loop_iterable(iterable):
    while True:
        yield from iterable
        
        
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def compute_kl(Xs,Xt,discr,optimizer,optim = True,nb_iter=10,device='cuda'):
    """
    Xt are the prior samples
    """
    batch_size = 2000
    # we work with without optimizing sample representation
    source_loader = create_data_loader(Xs.detach(), batch_size= batch_size)
    target_loader = create_data_loader(Xt, batch_size = batch_size)
    if optim == True:
    # learning the discriminator function 
        for epoch in range(nb_iter):
                S_batches = loop_iterable(source_loader)
                T_batches = loop_iterable(target_loader)
        
                iterations = len(source_loader)
                total_loss = 0
                for i in range(iterations):
                    source_s = next(S_batches)[0]
                    source_t = next(T_batches)[0]
        
                    out_s = discr(source_s.to(device))
                    out_t = discr(source_t.to(device))
                    
                    loss = - (torch.log(torch.sigmoid(out_s)).mean() + torch.log(1-torch.sigmoid(out_t)).mean())  
                    total_loss += loss.item()
                    #print(out_s.shape,out_t.shape,loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                #print('kl',epoch,-total_loss)
    # computing the KL with fixed discriminator and optimizable sample
    somme = 0    
    set_requires_grad(discr, requires_grad=False)  
    source_loader = create_data_loader(Xs, batch_size= batch_size)  
    for data in source_loader:
        somme += discr(data[0]).sum()

    return somme/Xs.shape[0]


def PAC_SWD(first_samples, second_samples, num_projections, prior_samples, lr = 0.005, p=2,
            max_iter=10, power_p = True, optim_lam = False, device="cuda",
            method= "NN", approx_vmf = True, mu_0=None, kappa_0=None):
    n = first_samples.size(0)
    lam = n**(0.5)
    dim = first_samples.size(1)
    rho = unif.HypersphericalUniform(dim = dim - 1)
    thetas = rho.sample(shape = num_projections)
    
    pro = thetas.to(device)
    
    first_samples_detach = first_samples.detach().to(device)
    second_samples_detach = second_samples.detach().to(device)
    if method == "NN":
        f = TransformNet(dim).to(device)
        f_op = optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.)
        f_op.zero_grad()
        discriminator = Discriminator(dim).to(device)
        discrim_optim = optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=0.00001)

    if method == "vmf":
        mu_t = mu_0
        kappa_t = kappa_0
        kappa_t.requires_grad = True
        kappa_pos = kappa_t
        params = [kappa_t]
        f_op = optim.Adam(params, lr=lr)
        rho_0 = unif.HypersphericalUniform(dim=dim-1,device=device)
        
    if optim_lam == 'GD' or optim_lam == 'optimal':
        K = torch.cdist(first_samples, second_samples, p = 2.0)
        Delta = torch.max(K).detach().cpu().numpy()
        if optim_lam == 'GD':
            alpha = torch.tensor(0.5, dtype = torch.float, device = device, requires_grad = True)
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
            f_op.add_param_group({'params': [alpha]})
        
    for i in range(max_iter):
        if method == "NN":
            thetas = f(pro)
            set_requires_grad(discriminator,True)
            kl_val = compute_kl(thetas,prior_samples, discriminator, discrim_optim,device=device)
            set_requires_grad(discriminator,False)
            wasserstein_distance = sliced_wasserstein_distance(first_samples_detach, second_samples_detach, projections = thetas, num_projections = num_projections, p=p, power_p = power_p, device=device)
            
        elif method == "vmf":
            kappa_pos.data.clamp_(min=1e-3)
            if kappa_t.data < 1e-3:
                print('Kappa is less than 1e-3, be careful!')
                rho_t = unif.HypersphericalUniform(dim=dim-1)
                thetas = rho_t.sample(shape=num_projections).to(device)
            else:
                rho_t = vmf.VonMisesFisher(mu_t / torch.norm(mu_t), kappa_pos, approx=approx_vmf)
                thetas = rho_t.rsample(shape=num_projections).to(device)
            wasserstein_distance = sliced_wasserstein_distance(first_samples_detach, second_samples_detach,
                                                               projections=thetas, num_projections=num_projections,
                                                               p=p, power_p = power_p, device=device)
            kl_val = torch.distributions.kl.kl_divergence(rho_t, rho_0).mean()
        
        if optim_lam == 'GD':
            reg = (kl_val + 4.6) / lam + lam * (Delta**(2*p))/(4*n)
        else:
            reg = kl_val / lam
        loss = - wasserstein_distance + reg
        if i % 100 == 0:
            print("Iteration {}".format(i))
            print("\t SW:{}".format(wasserstein_distance))
            print("\t Regularization:{}".format(reg))
        
        f_op.zero_grad()
        loss.backward()#retain_graph = True)
        f_op.step()
        if optim_lam =='GD':
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
        elif optim_lam == 'optimal':
            C_2 = Delta**(2*p)/4
            if method =="NN":
                K_1 = compute_kl(thetas,prior_samples, discriminator, discrim_optim, optim=False, device=device).detach().cpu().numpy().item()
            if method =="vmf":
                K_1 = kl_val.detach().cpu().numpy().item() 
            alpha_0 = 0.5 + 1/(2*np.log(n)) * np.log(2*(K_1+4.6)/C_2)
            lam = n**alpha_0
        else:
            pass
    if method == "NN":    
        thetas = f(pro).detach()
        wasserstein_distance = sliced_wasserstein_distance(first_samples, second_samples, projections = thetas, num_projections = num_projections, p=p, power_p = power_p, device=device)
        return wasserstein_distance, thetas #projections
    if method == "vmf":
        final_mu = (mu_t / torch.norm(mu_t)).detach()
        # final_kappa = kappa_pos.detach()
        kappa_pos.data.clamp_(min=1e-3)
        rho = vmf.VonMisesFisher(final_mu, kappa_pos, approx=approx_vmf)
        thetas = rho.rsample(shape=num_projections).to(device)
        wasserstein_distance = sliced_wasserstein_distance(first_samples, second_samples, projections = thetas, num_projections = num_projections, p=p, power_p = power_p, device=device)
        kl_val = torch.distributions.kl.kl_divergence(rho, rho_0).mean()
        reg = kl_val / lam
        bound = wasserstein_distance - kl_val 
        return wasserstein_distance, bound, (final_mu.detach(), kappa_pos.detach())



def pac_sliced_wasserstein_distance2(
    first_samples, second_samples, num_projections, prior_samples, p=2, max_iter=10, lam=1, power_p = True, optim_lam = False, device="cuda"):
    if optim_lam == 'GD' or optim_lam == 'optimal':
        K = torch.cdist(first_samples, second_samples, p = 2.0)
        Delta = torch.max(K).cpu().numpy()
        n = first_samples.size(1)
        lam = n**(0.5)
        if optim_lam == 'GD':
            alpha = torch.tensor(0.5, dtype = torch.float, device = device, requires_grad = True)
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
            f_alpha = optim.Adam([alpha], lr=0.01, betas=(0.5, 0.999),weight_decay=0.)
    else:
        n = first_samples.size(1)
        lam = n**(0.5)
        
    
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    
    f = TransformNet(embedding_dim).to(device)
    f_op = optim.Adam(f.parameters(), lr=0.001, betas=(0.5, 0.999),weight_decay=0.)
    discriminator = Discriminator(embedding_dim).to(device)
    discrim_optim =  optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999),weight_decay=0.00001)
    
    for i in range(max_iter):
        projections = f(pro)
        set_requires_grad(discriminator,True)
        
        kl_val = compute_kl(projections,prior_samples, discriminator, discrim_optim,device=device)
        #if optim_lam == 'GD':
        #    lam_numpy = torch.clone(lam).detach().cpu().numpy().item()
        #    
        #else:
        #    lam_numpy = lam
        
        reg = kl_val / lam
        #reg = kl_val / lam_numpy
        set_requires_grad(discriminator,False)
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]- torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        if power_p == True:
            wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1).mean()
        if power_p == False:
            wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p) #added the mean
        loss = reg - wasserstein_distance
        #print('lamb before', lam)
        f_op.zero_grad()
        loss.backward(retain_graph = True)
        f_op.step()
        #print('lamb after', lam)
        if optim_lam == 'GD':
            f_alpha.zero_grad()
            #set_requires_grad(discriminator,False)
            KL = compute_kl(projections,prior_samples, discriminator, discrim_optim, optim= False, device=device).detach().cpu().numpy().item()
            
            loss_lamb = lam * (Delta**(2*p))/(4*n) + (KL+4.6)/lam    #4.6 = ln(10**2)
            #print('alpha before', alpha)
            f_alpha.zero_grad()
            loss_lamb.backward()
            f_alpha.step()
            #print('alpha after', alpha)
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
        if optim_lam == 'optimal':
            C_2 = Delta**(2*p)/4
            K_1 = compute_kl(projections,prior_samples, discriminator, discrim_optim, optim=False, device=device).detach().cpu().numpy().item()
            alpha_0 = 0.5 + 1/(2*np.log(n)) * np.log(2*(K_1+4.6)/C_2)
            lam = n**alpha_0
        
    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    if power_p == True:
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1).mean()
    if power_p == False:
        wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance,projections


#def pac_sliced_wasserstein_distance(
#    first_samples, second_samples, num_projections, f, f_op, prior_samples, discriminator,discrim_optim,p=2, max_iter=10, lam=1, optim_lam = False, device="cuda"):
#    if optim_lam == 'GD' or optim_lam == 'optimal':
#        K = torch.cdist(first_samples, second_samples, p = 2.0)
#        Delta = torch.max(K).requires_grad_(False)
#        #print(Delta)
#        n = first_samples.size(1)
#        if optim_lam == 'GD':
#            lamm = n**(-0.5)
#            lamb = torch.tensor(lamm, requires_grad = True).to(device)
#            
#            f_op.add_param_group({'params':lamb})
#            #lamb = nn.Parameter(lamm*torch.ones(1))
#            #lamb = torch.ones((1), requires_grad=True, device=device) * lamm
#            #lamb_optim = optim.Adam(lamb, lr=0.001, betas=(0.5, 0.999),weight_decay=0.)
#    embedding_dim = first_samples.size(1)
#    pro = rand_projections(embedding_dim, num_projections).to(device)
#    #proj_prior = rand_projections(embedding_dim, num_projections).to(device)
#    discriminator=discriminator.to(device)
#    first_samples_detach = first_samples.detach()
#    second_samples_detach = second_samples.detach()
#    # learning the best distribution for max SWD under KL regularization
#    # we actually learn a pushforward
#    for _ in range(max_iter):
#        projections = f(pro)
#        # computing KL on the distribution of pushed projections vs projection
#        # samples from the prior using an adversarial approach
#        #discr = Discriminator(embedding_dim)
#        #optimizer = optim.Adam(discr.parameters(),lr = 0.001,weight_decay =0.001)
#        set_requires_grad(discriminator,True)
#
#        kl_val = compute_kl(projections,prior_samples, discriminator, discrim_optim,device=device)
#        if optim_lam == 'GD':
#            reg = kl_val / (lamb.cpu().detach().numpy()) 
#            lamb.requires_grad_(False)
#        else:
#            reg = kl_val/ lam
#        set_requires_grad(discriminator,False)
#        
#        # computing SWD 
#        #set_requires_grad(projections,True)
#        #projections.requires_grad_(True)
#        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
#        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
#        wasserstein_distance = torch.abs(
#            (
#                torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
#                - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
#            )
#        )
#        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
#        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
#        loss = reg - wasserstein_distance
#        
#        f_op.zero_grad()
#        loss.backward()
#        f_op.step()
#        #projections.requires_grad_(False)
#        if optim_lam == 'GD':
#            set_requires_grad(discriminator,False)
#            lamb.requires_grad_(True)
#            KL = np.asscalar(compute_kl(projections,prior_samples, discriminator, discrim_optim,optim= False,device=device).detach().cpu().numpy())
#            #KL.requires_grad_(False).detach()
#            loss_lamb = lamb * Delta**(2*p)/(4*n) + (KL+4.6)/lamb    #4.6 = ln(10**2)
#            f_op.zero_grad()
#            loss_lamb.backward()
#            f_op.step()
#        if optim_lam == 'optimal':
#            C_2 = Delta**(2*p)/4
#            C_2 = C_2.cpu().numpy()
#            K_1 = np.asscalar(compute_kl(projections,prior_samples, discriminator, discrim_optim,optim=False,device=device).detach().cpu().numpy())
#            alpha_0 = 0.5 + 1/(2*np.log(n)) * np.log(2*(K_1+4.6)/C_2)
#            #print('alpha',alpha_0)
#            lam = n**alpha_0
#            #print('lam',lam)
#            
#            #print(lamb)
#        #print(reg.item(),loss.item())
#    
#    projections = f(pro)
#    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
#    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
#    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
#    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
#    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
#
#    return wasserstein_distance,projections


def PAC_SWD_before_adding_vmf(first_samples, second_samples, num_projections, prior_samples, lr = 0.05, p=2, max_iter=10, power_p = True, optim_lam = False, device="cuda", method= "NN"):

    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    #print(first_samples.shape,second_samples.shape)

    f = TransformNet(embedding_dim).to(device)
    f_op = optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.999),weight_decay=0.)
    discriminator = Discriminator(embedding_dim).to(device)
    discrim_optim =  optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.999),weight_decay=0.00001)
    
    if optim_lam == 'GD' or optim_lam == 'optimal':
        K = torch.cdist(first_samples, second_samples, p = 2.0)
        Delta = torch.max(K).detach().cpu().numpy()
        n = first_samples.size(1)
        lam = n**(0.5)
        if optim_lam == 'GD':
            alpha = torch.tensor(0.5, dtype = torch.float, device = device, requires_grad = True)
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
            #f_alpha = optim.Adam([alpha], lr=0.01, betas=(0.5, 0.999),weight_decay=0.)
            f_op.add_param_group({'params': [alpha]})
    else:
        n = first_samples.size(1)
        lam = n**(0.5)
    for i in range(max_iter):
        projections = f(pro)
        set_requires_grad(discriminator,True)
        kl_val = compute_kl(projections,prior_samples, discriminator, discrim_optim,device=device)
        set_requires_grad(discriminator,False)
        
        if optim_lam == 'GD':
            reg = (kl_val+4.6) / lam + lam * (Delta**(2*p))/(4*n)
            #print((kl_val + 4.6)/lam)
        else:
            reg = (kl_val + 4.6) / lam
        
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph = True)
        f_op.step()
        if optim_lam =='GD':
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)

        if optim_lam == 'optimal':
            C_2 = Delta**(2*p)/4
            K_1 = compute_kl(projections,prior_samples, discriminator, discrim_optim, optim=False, device=device).detach().cpu().numpy().item()
            alpha_0 = 0.5 + 1/(2*np.log(n)) * np.log(2*(K_1+4.6)/C_2)
            lam = n**alpha_0
        
    projections = f(pro).detach()
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)

    return wasserstein_distance, projections



def PAC_SWD_vmf(first_samples, second_samples, num_projections, prior_samples, mu_t,kappa_t,lr = 0.001, p=2, max_iter=10, power_p = True, optim_lam = False, device="cuda", approx_vmf = True):
    # if method == "vmf":
    #     device='cpu'
    method = "vmf"
    n = first_samples.size(0)
    lam = n**(0.5)
    dim = first_samples.size(1)
    mu = torch.zeros(dim)
    kappa = torch.tensor([1])
    rho = unif.HypersphericalUniform(dim = dim - 1)
    thetas = rho.sample(shape = num_projections)
    
    pro = thetas.to(device)
    
    first_samples_detach = first_samples.detach().to(device)
    second_samples_detach = second_samples.detach().to(device)
    if method == "NN":
        f = TransformNet(dim).to(device)
        f_op = optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.)
        f_op.zero_grad()
        discriminator = Discriminator(dim).to(device)
        discrim_optim = optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=0.00001)
        
    if method == "vmf":
        
        #m1 = torch.mean(first_samples_detach.clone().cpu(), dim=0)
        #m2 = torch.mean(second_samples_detach.clone().cpu(), dim=0)
        #mu_t = torch.tensor(m1-m2, requires_grad=True, dtype=torch.float).to(device)
        # mu_t = torch.zeros(dim, requires_grad=True)#,dtype=torch.float )
        
        #mu_t = torch.randn(dim, requires_grad=True,device=device)
        #kappa_t = torch.tensor([dim/2], requires_grad=True, dtype=torch.float,device=device) #torch.ones(1, requires_grad=True, dtype=torch.float)
        kappa_pos = torch.exp(kappa_t)
        params = [mu_t, kappa_t]
        f_op = optim.Adam(params, lr=lr)
        f_op.zero_grad()
        rho_0 = unif.HypersphericalUniform(dim=dim-1,device=device)
        #print(kappa_t)
        
    if optim_lam == 'GD' or optim_lam == 'optimal':
        K = torch.cdist(first_samples, second_samples, p = 2.0)
        Delta = torch.max(K).detach().cpu().numpy()
        if optim_lam == 'GD':
            alpha = torch.tensor(0.5, dtype = torch.float, device = device, requires_grad = True)
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
            f_op.add_param_group({'params': [alpha]})
        
    for i in range(max_iter):
        
        if method == "NN":
            thetas = f(pro)
            set_requires_grad(discriminator,True)
            kl_val = compute_kl(thetas,prior_samples, discriminator, discrim_optim,device=device)
            set_requires_grad(discriminator,False)
            wasserstein_distance = sliced_wasserstein_distance(first_samples_detach, second_samples_detach, projections = thetas, num_projections = num_projections, p=p, power_p = power_p, device=device)
            #encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
            #distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
            #wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)
            
        elif method == "vmf":
            #kappa_t = kappa_t.data.clamp_(min=1e-4, max = 1000)
            #kappa_t = torch.clamp(kappa_t, min = 1.5e-3)
            #if kappa_t.data == 0:
            #print(kappa_t)
            kappa_t.data.clamp_(min=1e-3)
            if kappa_t.data < 1e-3:
                print('Kappa is less than 1e-3, be careful!')
                #kappa_t.data.clamp_(min=1e-3)
                #kappa_t = torch.abs(kappa_t.clone())
                rho_t = unif.HypersphericalUniform(dim=dim-1)
                thetas = rho_t.sample(shape=num_projections).to(device)
                #kappa_t = kappa_t *10
            else:
                #print('good to be here')
                rho_t = vmf.VonMisesFisher(mu_t / torch.norm(mu_t), kappa_pos, approx=approx_vmf)  # torch.abs(kappa_t))
                thetas = rho_t.rsample(shape=num_projections).to(device)
            wasserstein_distance = sliced_wasserstein_distance(first_samples_detach, second_samples_detach,
                                                               projections=thetas, num_projections=num_projections,
                                                               p=p, power_p = power_p, device=device)
            #rho_t = vmf.VonMisesFisher(mu_t, kappa_t)
            kl_val = torch.distributions.kl.kl_divergence(rho_t, rho_0).mean()
            #print('kl',kl_val)
        
        if optim_lam == 'GD':
            reg = (kl_val + 4.6) / lam + lam * (Delta**(2*p))/(4*n)
        else:
            reg = kl_val / lam
        
        f_op.zero_grad() #to uncomment if issues of graph
        loss = - wasserstein_distance + reg
        
       # if i % 50 == 0:
       #     print('kl',kl_val)
       #     print("method {}, value of reg {} and wass {}".format(method, reg,wasserstein_distance))
       #     if method == 'vmf':
       #         print(kappa_t)
        
        loss.backward(retain_graph = True) 
        f_op.step()
        if optim_lam =='GD':
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
        elif optim_lam == 'optimal':
            C_2 = Delta**(2*p)/4
            if method =="NN":
                K_1 = compute_kl(thetas,prior_samples, discriminator, discrim_optim, optim=False, device=device).detach().cpu().numpy().item()
            if method =="vmf":
                K_1 = kl_val.detach().cpu().numpy().item() 
            alpha_0 = 0.5 + 1/(2*np.log(n)) * np.log(2*(K_1+4.6)/C_2)
            lam = n**alpha_0
        else:
            pass
    if method == "NN":    
        thetas = f(pro).detach()
        wasserstein_distance = sliced_wasserstein_distance(first_samples, second_samples, projections = thetas, num_projections = num_projections, p=p, power_p = power_p, device=device)
    if method == "vmf":
        rho = vmf.VonMisesFisher(mu_t.detach(), kappa_pos.detach(), approx = approx_vmf)
        thetas = rho.rsample(shape=num_projections).to(device)
        wasserstein_distance = sliced_wasserstein_distance(first_samples, second_samples, projections = thetas, num_projections = num_projections, p=p, power_p = power_p, device=device)
        #print(kappa_t)
    #encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    #distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    #wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)
    
    return wasserstein_distance, thetas #projections


def PAC_SWD_nn(first_samples, second_samples, num_projections, prior_samples, f,f_op
                      ,discriminator,discrim_optim, lr = 0.05, p=2, max_iter=10, power_p = True, optim_lam = False, device="cuda", method= "NN"):

    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    #print(first_samples.shape,second_samples.shape)

    # #f = TransformNet(embedding_dim).to(device)
    # #f_op = optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.999),weight_decay=0.)
    # discriminator = Discriminator(embedding_dim).to(device)
    # discrim_optim =  optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.999),weight_decay=0.00001)
    
    if optim_lam == 'GD' or optim_lam == 'optimal':
        K = torch.cdist(first_samples, second_samples, p = 2.0)
        #Delta = torch.max(K).detach().cpu().numpy()
        Delta = torch.max(K).detach()

        n = first_samples.size(1)
        lam = n**(0.5)
        if optim_lam == 'GD':
            alpha = torch.tensor(0.5, dtype = torch.float, device = device, requires_grad = True)
            lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
            #f_alpha = optim.Adam([alpha], lr=0.01, betas=(0.5, 0.999),weight_decay=0.)
            f_op.add_param_group({'params': [alpha]})
    else:
        n = first_samples.size(1)
        lam = n**(0.5)
    for i in range(max_iter):
        projections = f(pro)
        set_requires_grad(discriminator,True)
        kl_val = compute_kl(projections,prior_samples, discriminator, discrim_optim,device=device)
        set_requires_grad(discriminator,False)
        
        if optim_lam == 'GD':
            reg = (kl_val+4.6) / lam + lam * (Delta**(2*p))/(4*n)
            #print((kl_val + 4.6)/lam)
        else:
            reg = (kl_val + 4.6) / lam
        
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)
        loss = reg - wasserstein_distance
        #alpha_old = alpha.detach()
        f_op.zero_grad()
        loss.backward(retain_graph = True)
        f_op.step()
        if optim_lam =='GD':
            #lam = torch.pow(torch.tensor(n, dtype = torch.float, device = device, requires_grad = False), alpha)
            lam = torch.pow(n, alpha)
        if optim_lam == 'optimal':
            C_2 = Delta**(2*p)/4
            K_1 = compute_kl(projections,prior_samples, discriminator, discrim_optim, optim=False, device=device).detach().cpu().numpy().item()
            alpha_0 = 0.5 + 1/(2*np.log(n)) * np.log(2*(K_1+4.6)/C_2)
            lam = n**alpha_0
        
    projections = f(pro).detach()
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = Wasserstein_1D(encoded_projections, distribution_projections, p = p, power_p = power_p)

    return wasserstein_distance, projections

=======
    
    return wasserstein_distance, thetas #projections
>>>>>>> 7fb258a3b92c0015e4628556b942caae8d685b20
