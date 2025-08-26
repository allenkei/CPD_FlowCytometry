import torch
import numpy as np
from math import comb
import scipy.stats as st
import matplotlib.pyplot as plt

def evaluation(delta_mu, args, loglik, pen_iter, seq_iter, output_dir, half):

  if half:
    T = int(args.num_time/2)
    tau = T-1 # 50 time points - 1 = 49 differences
    true_CP = args.true_CP_half # [14,26,39]
    label = 'half'
  else:
    T = args.num_time
    tau = T-1 # 100 time points - 1 = 99 differences
    true_CP = args.true_CP_full # [26,51,76]
    label = 'full'

  
  delta_mu = delta_mu.cpu()
  t_change = (delta_mu - torch.median(delta_mu)) / torch.std(delta_mu)
  threshold = torch.mean(t_change) + torch.tensor(st.norm.ppf(0.9)) * torch.std(t_change)

  # store change points (exclude first and last 5)
  est_CP = []
  for i in range(tau):
    if t_change[i] > threshold and i >= 5 and i <= tau-5:
      est_CP.append(i)

  #print('est_CP (before min-spacing):', est_CP)

  # min-spacing
  end_i = 1
  while end_i < len(est_CP):
    prev = est_CP[end_i-1]
    this = est_CP[end_i]

    if this - prev > 5: # spacing > 5
      end_i += 1
    else:
      selection = [prev, this]
      to_remove = selection[torch.argmin(delta_mu[selection])] # keep the highest
      est_CP.remove(to_remove)

  #print('est_CP (after min-spacing):', est_CP)

  # there are 100 time points, 99 differences
  # if the 25th difference (index 24) > threshold, the actual time point is 26, so index 24+2=26
  est_CP = [cp + 2 for cp in est_CP] # CP as actual time point
  num_CP = len(est_CP)

  #print('est_CP (in actual time):', est_CP)

  # intervals
  gt_CP_all = [1] + true_CP + [T + 1]
  est_CP_all = [1] + est_CP + [T + 1]
  gt_list = [range(gt_CP_all[i-1], gt_CP_all[i]) for i in range(1, len(gt_CP_all))]
  est_list = [range(est_CP_all[i-1], est_CP_all[i]) for i in range(1, len(est_CP_all))]

  #print('est_CP:', est_CP)
  #print('gt_list:', gt_list)
  #print('est_list:', est_list)

  if num_CP == 0:
    dist_est_gt = float('inf')
    dist_gt_est = float('-inf')
    covering_metric = 0
  else:
    # calculate the 2 one-sided distance
    holder_est_gt = []
    for i in true_CP:
      dist_diff_est_gt = [abs(j-i) for j in est_CP]
      holder_est_gt.append(min(dist_diff_est_gt))
    dist_est_gt = max(holder_est_gt)

    holder_gt_est = []
    for i in est_CP:
      dist_diff_gt_est = [abs(j-i) for j in true_CP]
      holder_gt_est.append(min(dist_diff_gt_est))
    dist_gt_est = max(holder_gt_est)

    # calculate covering metric
    covering_metric = 0
    for A in gt_list:
      jaccard = []
      for A_prime in est_list:
        jaccard.append( len(set(A).intersection(set(A_prime))) / len(set(A).union(set(A_prime))) )
      covering_metric += len(A) * max(jaccard)
    covering_metric /= tau + 1

  abs_error = abs(num_CP - len(true_CP))


  # save final visualization
  plt.plot(np.arange(T-1), t_change); plt.xticks(true_CP)
  plt.axhline(y=threshold.numpy(), color='r', linestyle='-')
  plt.savefig( output_dir + '/{}_mu_CP_seq{}pen{}.png'.format(label,seq_iter,pen_iter) ) 
  plt.close()
  
  output = [abs_error, dist_est_gt, dist_gt_est, covering_metric, loglik, threshold.numpy().item(), est_CP]
  return output








