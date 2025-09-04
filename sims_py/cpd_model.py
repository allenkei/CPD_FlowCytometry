import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

##################
# ARGUMENT PARSER
##################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', default=1, type=int) 
    parser.add_argument('--x_dim', default=3, type=int) 
    parser.add_argument('--y_dim', default=3, type=int)
    parser.add_argument('--K_dim', default=3, type=int)
    parser.add_argument('--output_layer', nargs='+', type=int, default=[128,128]) 
    parser.add_argument('--num_samples', default=100, type=int) 
    parser.add_argument('--num_time', default=200, type=int)
    parser.add_argument('--penalties', nargs='+', type=float, default=[0.01,0.1,0.5,1,10,100])
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--decoder_lr', default=0.01, type=float)
    parser.add_argument('--langevin_K', default=100, type=int)
    parser.add_argument('--langevin_s', default=0.1, type=float)
    parser.add_argument('--kappa', default=0.1, type=float)
    parser.add_argument('--nu_iteration', default=20, type=int)
    parser.add_argument('--decoder_iteration', default=20, type=int)
    parser.add_argument('--loss_thr', default=1e-5, type=float)
    parser.add_argument('--iter_thr', default=5, type=int)
    parser.add_argument('--true_CP_full', nargs='+', type=int, default=[51,101,151])
    parser.add_argument('--signif_level', default=0.975, type=float)
    args, _ = parser.parse_known_args()
    return args

##################
# LOSS FUNCTION  #
##################
def mixture_of_gaussians_loss(y, pi, mean, sigma):
    """
    y:     (N, D)
    pi:    (N, K)
    mean:  (N, K, D)
    sigma: (N, K, D)
    """
    N, K, D = mean.shape
    y_expand = y.unsqueeze(1).expand(-1, K, -1)
    diff = y_expand - mean
    mahal = torch.sum((diff**2) / (sigma + 1e-8), dim=2)  # (N, K)
    log_det = torch.sum(torch.log(sigma + 1e-8), dim=2)   # (N, K)
    log_prob = -0.5 * (D * np.log(2*np.pi) + log_det + mahal)
    weighted = pi * torch.exp(log_prob)
    total_prob = torch.sum(weighted, dim=1)
    nll = -torch.sum(torch.log(total_prob + 1e-12))
    return nll

##################
# MODEL DEFINITION
##################
class CPD(nn.Module):
    def __init__(self, args, half):
        super(CPD, self).__init__()
        self.d = args.z_dim
        self.p = args.x_dim
        self.K = args.K_dim
        self.D = args.y_dim
        self.T = int(args.num_time/2) if half else args.num_time

        self.l1 = nn.Linear(self.d + self.p, args.output_layer[0])
        self.l2 = nn.Linear(args.output_layer[0], args.output_layer[1])
        self.l3_pi    = nn.Linear(args.output_layer[1], self.K)
        self.l3_mean  = nn.Linear(args.output_layer[1], self.K * self.D)
        self.l3_sigma = nn.Linear(args.output_layer[1], self.K * self.D)

    def forward(self, x, z):
        output = torch.cat([x, z], dim=1)
        output = self.l1(output).tanh()
        output = self.l2(output).tanh()
        pi = self.l3_pi(output).softmax(dim=1)
        mean = self.l3_mean(output).reshape(-1, self.K, self.D)
        sigma = F.softplus(self.l3_sigma(output)).reshape(-1, self.K, self.D) + 1e-6
        return pi, mean, sigma

    def infer_z(self, x, z, y, mu_repeat, args):
        for k in range(args.langevin_K):
            z = z.detach().clone().requires_grad_(True)
            pi, mean, sigma = self.forward(x, z)
            nll = mixture_of_gaussians_loss(y, pi, mean, sigma)
            z_grad_nll = torch.autograd.grad(nll, z)[0]
            noise = torch.randn_like(z).to(z.device)
            z = z + args.langevin_s * (-z_grad_nll - (z - mu_repeat)) \
                  + torch.sqrt(torch.tensor(2.0 * args.langevin_s, device=z.device)) * noise
        return z.detach()

##################
# TRAINING LOGIC #
##################
def learn_one_seq_penalty(args, x_input_train, y_input_train,
                          x_input_test, y_input_test, penalty, half):
    """
    One sequence learning with given penalty.
    Inputs are expected to be torch tensors already moved to CPU/GPU.
    """
    m = args.num_samples
    kappa = args.kappa
    d = args.z_dim
    T = x_input_train.shape[0] // args.num_samples   

    print(f"\n[INFO] Penalty {penalty}")


    dev = x_input_train.device


    ones_col = torch.ones(T, 1, device=dev)
    X = torch.zeros(T, T-1, device=dev)
    i, j = torch.tril_indices(T, T-1, offset=-1)
    X[i, j] = 1  # Group Fused Lasso

    mu = torch.zeros(T, d, device=dev)
    nu = torch.zeros(T, d, device=dev)
    w  = torch.zeros(T, d, device=dev)

    mu_old = mu.clone()
    nu_old = nu.clone()


    model = CPD(args, half=False).to(dev)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.decoder_lr)

    old_loss_train = -float('inf')
    stopping_count = 0
    early_stopping = False

    for learn_iter in range(args.epoch):

        mu_repeat = mu.repeat_interleave(m, dim=0)
        init_z = torch.randn(T*m, d, device=dev)
        sampled_z_all = model.infer_z(x_input_train, init_z, y_input_train, mu_repeat, args)


        expected_z = sampled_z_all.clone().reshape(T, m, d).mean(dim=1)
        mu = (expected_z + kappa * (nu - w)) / (1.0 + kappa)
        mu = mu.detach().clone()


        for _ in range(args.decoder_iteration):
            optimizer.zero_grad()
            pi, mean, sigma = model(x_input_train, sampled_z_all)
            loss_train = mixture_of_gaussians_loss(y_input_train, pi, mean, sigma) / m
            loss_train.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()


        gamma = nu[0, :].unsqueeze(0)
        beta = torch.diff(nu, dim=0)

        for _ in range(args.nu_iteration):
            for t in range(T-1): 
                beta_without_t = beta.clone()
                X_without_t = X.clone()
                beta_without_t[t, :] = 0
                X_without_t[:, t] = 0

                bt = kappa * torch.mm(
                    X[:, t].unsqueeze(0),
                    mu + w - torch.mm(ones_col, gamma) - torch.mm(X_without_t, beta_without_t)
                )
                bt_norm = torch.norm(bt, p=2)

                if bt_norm < penalty:
                    beta[t, :] = 0
                else:
                    beta[t, :] = (1 / (kappa * torch.norm(X[:, t], p=2)**2)) \
                                  * (1 - penalty/bt_norm) * bt
            gamma = torch.mean(mu + w - torch.mm(X, beta), dim=0).unsqueeze(0)

        nu = torch.mm(ones_col, gamma) + torch.mm(X, beta)
        w = mu - nu + w


        loss_relative_diff = torch.abs((loss_train - old_loss_train) / old_loss_train) \
                             if old_loss_train != -float('inf') else torch.tensor(1.0)
        old_loss_train = loss_train.clone()
        mu_old = mu.clone()
        nu_old = nu.clone()

        if loss_relative_diff < args.loss_thr:
            stopping_count += 1
        else:
            stopping_count = 0

        if stopping_count >= args.iter_thr:
            early_stopping = True


        if early_stopping or (learn_iter+1) == args.epoch:
            if half:
                return loss_train.item(), penalty
            else:
                delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1).cpu().numpy()
                return evaluation(delta_mu, args)

##################
# EVALUATION LOGIC
##################
def evaluation(delta_mu, args):
    """delta_mu should be a numpy array (on CPU)."""
    T = args.num_time
    tau = T - 1
    true_CP = args.true_CP_full

    delta_mu = torch.tensor(delta_mu, dtype=torch.float32)
    t_change = (delta_mu - torch.median(delta_mu)) / torch.std(delta_mu)
    threshold = torch.mean(t_change) + torch.tensor(st.norm.ppf(args.signif_level)) * torch.std(t_change)

    est_CP = []
    for i in range(tau):
        if t_change[i] > threshold and i >= 5 and i <= tau-5:
            est_CP.append(i)
    # min-spacing
    end_i = 1
    while end_i < len(est_CP):
        prev, this = est_CP[end_i-1], est_CP[end_i]
        if this - prev <= 5:
            selection = [prev, this]
            to_remove = selection[torch.argmin(delta_mu[selection])]
            est_CP.remove(to_remove)
        else:
            end_i += 1
    est_CP = [cp+2 for cp in est_CP]

    # metrics
    num_CP = len(est_CP)
    if num_CP == 0:
        dist_est_gt, dist_gt_est, covering_metric = float('inf'), float('-inf'), 0
    else:
        holder_est_gt = [min([abs(j-i) for j in est_CP]) for i in true_CP]
        dist_est_gt = max(holder_est_gt)
        holder_gt_est = [min([abs(j-i) for j in true_CP]) for i in est_CP]
        dist_gt_est = max(holder_gt_est)
        gt_CP_all  = [1] + true_CP + [T+1]
        est_CP_all = [1] + est_CP + [T+1]
        gt_list  = [range(gt_CP_all[i-1], gt_CP_all[i]) for i in range(1, len(gt_CP_all))]
        est_list = [range(est_CP_all[i-1], est_CP_all[i]) for i in range(1, len(est_CP_all))]
        covering_metric = 0
        for A in gt_list:
            jaccard = [len(set(A)&set(Ap))/len(set(A)|set(Ap)) for Ap in est_list]
            covering_metric += len(A)*max(jaccard)
        covering_metric /= tau+1
    abs_error = abs(num_CP - len(true_CP))

    # plot
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(np.arange(2, T+1), t_change, label="Δμ", color="#172478")
    for cp in est_CP: ax.axvline(x=cp, color="#B8860B", linestyle="--", lw=2.5, label="Estimated CP")
    for cp in true_CP: ax.axvline(x=cp, color="red", linestyle="-", lw=1, label="True CP")
    ax.axhline(y=threshold, color="#9467bd", linestyle="--", lw=1.5, alpha=0.8, label="Threshold")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels_leg, handles)).values(), dict(zip(labels_leg, handles)).keys(), loc="upper right")

    return abs_error, dist_est_gt, dist_gt_est, covering_metric, threshold.item(), est_CP, fig

##################
# UTILS
##################
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.05, 0.05)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.05, 0.05)
