import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import distributional_sliced_wasserstein_distance
from swd_pac import PAC_SWD
import matplotlib.pyplot as plt
import pickle
import os

from vmf_utils import hyperspherical_uniform as unif, von_mises_fisher

torch.manual_seed(10)
plt.rcParams.update({'font.size': 18})


def generate_data(n, d, type):
    if type == "uniform":
        X = torch.FloatTensor(n, d).uniform_(0, 5)
        Y = torch.FloatTensor(n, d).uniform_(0, 5)
    elif type == "gaussian":
        Id = torch.eye(d)
        mean = torch.zeros(d)
        Sigma = torch.rand(d, d)
        Sigma = torch.mm(Sigma, Sigma.t())
        Sigma.add_(Id)
        X = MultivariateNormal(mean, Sigma).rsample(sample_shape=torch.Size([n]))
        Y = MultivariateNormal(mean, Sigma).rsample(sample_shape=torch.Size([n]))
    else:
        return "Not implemented."
    return X, Y


def compute_vmf(kappas, n_samples, dims, n_runs, n_proj, order, datatype):
    vmf_sw = torch.zeros(size=(len(kappas), len(dims), len(n_samples), n_runs))
    for ki in range(len(kappas)):
        kappa = torch.tensor([kappas[ki]], dtype=torch.float)
        for i in range(len(dims)):
            d = dims[i]
            print("Dimension = {}".format(d))
            mu = torch.randn(d)
            mu /= torch.norm(mu)
            for j in range(len(n_samples)):
                for nr in range(n_runs):
                    print("\t Run {}".format(nr+1))
                    X, Y = generate_data(n=n_samples[j], d=d, type=datatype)
                    vmf_sw[ki, i, j, nr] = PAC_SWD(
                        X, Y, n_proj, prior_samples=None, p=2, max_iter=0, power_p=True, 
                        optim_lam=False, device=device, method="vmf", approx_vmf=False, mu_0=mu, kappa_0=kappa
                    )
                    with open(type + "_vmf_sw", "wb") as f:
                        pickle.dump(vmf_sw, f, pickle.HIGHEST_PROTOCOL)
            plt.plot(n_samples, torch.abs(vmf_sw[ki, i]).mean(axis=1), label="d={}, kappa={}".format(d, kappas[ki]))
    plt.legend()
    plt.show()


def exp_1():
    dims = [5, 10, 50, 100]
    n_samples = [50, 100, 500, 1000, 2000]
    n_proj = 1000
    type = "gaussian"
    kappas = [0.1, 1, 10, 100]
    n_runs = 30

    compute_vmf(kappas=kappas, n_samples=n_samples, dims=dims, n_runs=n_runs, n_proj=n_proj, order=2, datatype=type)

    file = open(type + "_vmf_sw", "rb")
    vmf_sw = pickle.load(file)

    # Study influence of kappas (impact on KL)
    for di in range(len(dims)):
        dim = dims[di]
        plt.figure()
        for ki in range(len(kappas)):
            kappa = kappas[ki]
            plt.plot(n_samples, vmf_sw[ki, di].mean(axis=1), label=r"$d={}, \kappa={}$".format(dim, kappa))
            plt.fill_between(n_samples, vmf_sw[ki, di].quantile(q=0.1, axis=1), vmf_sw[ki, di].quantile(q=0.9, axis=1),
                             alpha=0.2)
        plt.xlabel(r"$n$")
        plt.ylabel(r"$SW_2^2(\mu_n, \nu_n ; vMF_{\kappa})$")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join("results_exp_synth", str(type) + "_vmf_sw_d=" + str(dim) + ".pdf"))

    # Study influence of dimension (impact on diameter)
    for ki in range(len(kappas)):
        kappa = kappas[ki]
        plt.figure()
        for di in range(len(dims)):
            dim = dims[di]
            plt.plot(n_samples, vmf_sw[ki, di].mean(axis=1), label=r"$d={}, \kappa={}$".format(dim, kappa))
            plt.fill_between(n_samples, vmf_sw[ki, di].quantile(q=0.1, axis=1), vmf_sw[ki, di].quantile(q=0.9, axis=1),
                             alpha=0.2)
        plt.xlabel(r"$n$")
        plt.ylabel(r"$SW_2^2(\mu_n, \nu_n ; vMF_{\kappa})$")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join("results_exp_synth",  str(type) + "_vmf_sw_kap=" + str(kappa) + ".pdf"))


def exp_2(datatype, factors_y, dims, lbds, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    n_samples = 500
    n_test = 2000
    n_proj = 1000
    max_iter = 1500
    device = "cuda"
    n_runs = 10

    dsw_vmf_train = torch.zeros(size=(len(dims), len(factors_y), len(lbds), n_runs))
    dsw_vmf_test = torch.zeros(size=(len(dims), len(factors_y), len(lbds), n_runs))
    bound_dsw = torch.zeros(size=(len(dims), len(factors_y), len(lbds), n_runs))
    
    pac_sw_vmf_train = torch.zeros(size=(len(dims), len(factors_y), n_runs))
    pac_sw_vmf_test = torch.zeros(size=(len(dims), len(factors_y), n_runs))
    bound_pac = torch.zeros(size=(len(dims), len(factors_y), n_runs))

    for di in range(len(dims)):
        dim = dims[di]
        print("Dimension: {}".format(dim))
        Id = torch.eye(dim)
        mean_x = torch.zeros(dim)
        mu = torch.randn(dim).to(device)
        mu /= torch.norm(mu)
        kappa = torch.tensor([1], dtype=torch.float).to(device)
        if datatype == "gaussian":
            # Generate random covariance matrix
            Sigma_k = torch.rand(dim, dim)
            Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
            Sigma_k.add_(torch.eye(dim))
        for fi in range(len(factors_y)):
            for nr in range(n_runs):
                print("Run {}".format(nr + 1))
                if datatype == "uniform":
                    X = torch.FloatTensor(n_samples, dim).uniform_(-1, 1)
                    Y = torch.FloatTensor(n_samples, dim).uniform_(-factors_y[fi], factors_y[fi])
                    Xtest = torch.FloatTensor(n_test, dim).uniform_(-1, 1)
                    Ytest = torch.FloatTensor(n_test, dim).uniform_(-factors_y[fi], factors_y[fi])
                elif datatype == "gaussian":
                    # Generate data
                    X = MultivariateNormal(mean_x, Sigma_k).rsample(sample_shape=torch.Size([n_samples]))
                    Y = MultivariateNormal(factors_y[fi] * torch.ones(dim), Sigma_k).rsample(sample_shape=torch.Size([n_samples]))
                    Xtest = MultivariateNormal(mean_x, Sigma_k).rsample(sample_shape=torch.Size([n_test]))
                    Ytest = MultivariateNormal(factors_y[fi] * torch.ones(dim), Sigma_k).rsample(sample_shape=torch.Size([n_test]))
                for li in range(len(lbds)):
                    print("Lambda {}".format(lbds[li]))
                    dsw_vmf_train[di, fi, li, nr], bound_dsw[di, fi, li, nr], (_, kappa_dsw) = distributional_sliced_wasserstein_distance(
                        X, Y, n_proj, f=None, f_op=None, p=2, power_p=True, max_iter=max_iter, lam=lbds[li], 
                        device=device, method="vMF", mu_0=mu.clone(), kappa_0=kappa.clone(), approx_vmf=False
                    )
                    with open(os.path.join(folder, "dsw_train"), "wb") as f:
                        pickle.dump(dsw_vmf_train, f, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(folder, "bound_dsw"), "wb") as f:
                        pickle.dump(bound_dsw, f, pickle.HIGHEST_PROTOCOL)
                    print("kappa dsw:{}".format(kappa_dsw))
                    
                    dsw_vmf_test[di, fi, li, nr], _, _ = distributional_sliced_wasserstein_distance(
                        Xtest, Ytest, n_proj, f=None, f_op=None, p=2, power_p=True,
                        max_iter=0, lam=lbds[li], device=device, method="vMF", mu_0=mu.clone(), kappa_0=kappa_dsw.clone()
                    )
                    
                    with open(os.path.join(folder, "dsw_test"), "wb") as f:
                        pickle.dump(dsw_vmf_test, f, pickle.HIGHEST_PROTOCOL)
                    print("DSW test:{}".format(dsw_vmf_test[di, fi, li, nr]))

                print("DSW done. PAC-SW begins...")
                pac_sw_vmf_train[di, fi, nr], bound_pac[di, fi, nr], (_, kappa_pac) = PAC_SWD(
                    X, Y, n_proj, prior_samples=None, p=2, max_iter=max_iter, power_p=True, 
                    optim_lam=False, device=device, method="vmf", approx_vmf=False, mu_0=mu.clone(), kappa_0=kappa.clone()
                )
                print("PAC-SW done.")
                print("kappa pac:{}".format(kappa_pac))
                
                with open(os.path.join(folder, "pac_sw_train"), "wb") as f:
                    pickle.dump(pac_sw_vmf_train, f, pickle.HIGHEST_PROTOCOL)
                    
                with open(os.path.join(folder, "bound_pac"), "wb") as f:
                    pickle.dump(bound_pac, f, pickle.HIGHEST_PROTOCOL)
                
                pac_sw_vmf_test[di, fi, nr], _, _ = PAC_SWD(
                    Xtest, Ytest, n_proj, prior_samples=None, p=2, max_iter=0, power_p=True,
                    optim_lam=False, device=device, method="vmf", approx_vmf=False, mu_0=mu.clone(), kappa_0=kappa_pac.clone()
                )
                print("PAC test:{}".format(pac_sw_vmf_test[di, fi, nr]))
                with open(os.path.join(folder, "pac_sw_test"), "wb") as f:
                    pickle.dump(pac_sw_vmf_test, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Generate Figure 1
    exp_1()

    # Generate Figure 2
    folder = "exp_2"
    datatype = "gaussian"
    factors_y = [2, 3, 4, 5]
    dims = [5, 20]
    lbds = [1, 10, 100, 1000]
    folder = os.path.join(folder, datatype)
    
    exp_2(datatype, factors_y, dims, lbds, folder)
    
    file = open(os.path.join(folder, "dsw_train"), "rb")
    dsw_vmf_train = pickle.load(file)

    file = open(os.path.join(folder, "dsw_test"), "rb")
    dsw_vmf_test = pickle.load(file)
    
    file = open(os.path.join(folder, "pac_sw_train"), "rb")
    pac_sw_vmf_train = pickle.load(file)

    file = open(os.path.join(folder, "pac_sw_test"), "rb")
    pac_sw_vmf_test = pickle.load(file)
    
    file = open(os.path.join(folder, "bound_pac"), "rb")
    bound_pac = pickle.load(file)
    
    file = open(os.path.join(folder, "bound_dsw"), "rb")
    bound_dsw = pickle.load(file)
    
    for di in range(len(dims)):
        plt.figure(figsize=(8,4))
        plt.grid()
        for li in range(len(lbds)):
            # Train set
            plt.plot(factors_y, dsw_vmf_train[di, :, li].detach().numpy().mean(axis=1), label=r"DSW train".format(lbds[li]), c="blue") #$\lambda={}$
            plt.fill_between(factors_y, dsw_vmf_train[di, :, li].detach().quantile(q=0.1, axis=1), dsw_vmf_train[di, :, li].detach().quantile(q=0.9, axis=1), alpha=0.2, facecolor="blue")
            # Test set
            plt.plot(factors_y, dsw_vmf_test[di, :, li].detach().numpy().mean(axis=1), label=r"DSW test".format(lbds[li]), ls="--", c="blue")
            plt.fill_between(factors_y, dsw_vmf_test[di, :, li].detach().quantile(q=0.1, axis=1), dsw_vmf_test[di, :, li].detach().quantile(q=0.9, axis=1), alpha=0.2, facecolor="blue")
            # Plot DSW bound
            plt.plot(factors_y, bound_dsw[di, :, li].detach().numpy().mean(axis=1), label=r"DSW bound".format(lbds[li]), ls="-.", c="orange")
            plt.fill_between(factors_y, bound_dsw[di, :, li].detach().quantile(q=0.1, axis=1), bound_dsw[di, :, li].detach().quantile(q=0.9, axis=1), alpha=0.2, facecolor="red")

        plt.plot(factors_y, pac_sw_vmf_train[di].detach().numpy().mean(axis=1), label="PAC-SW train", c="red")
        plt.fill_between(factors_y, pac_sw_vmf_train[di].detach().quantile(q=0.1, axis=1), pac_sw_vmf_train[di].detach().quantile(q=0.9, axis=1), alpha=0.2, facecolor="red")
        plt.plot(factors_y, pac_sw_vmf_test[di].detach().numpy().mean(axis=1), label="PAC-SW test", ls="--", c="red")
        plt.fill_between(factors_y, pac_sw_vmf_test[di].detach().quantile(q=0.1, axis=1), pac_sw_vmf_test[di].detach().quantile(q=0.9, axis=1), alpha=0.2, color="red")
                             
        # Plot PAC Bound
        plt.plot(factors_y, bound_pac[di].detach().numpy().mean(axis=1), label="PAC bound", ls="-.", c="green")
        plt.fill_between(factors_y, bound_pac[di].detach().quantile(q=0.1, axis=1), bound_pac[di].detach().quantile(q=0.9, axis=1), alpha=0.2, facecolor="green")
        plt.xlabel(r"$\gamma$")
        if di == 0:
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "d=" + str(dims[di]) + ".pdf"))
