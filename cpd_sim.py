import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import os
torch.set_printoptions(precision=5)



def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--z_dim', default=10, type=int) 
  parser.add_argument('--x_dim', default=5, type=int) 
  parser.add_argument('--y_dim', default=3, type=int)
  parser.add_argument('--K_dim', default=2, type=int)
  parser.add_argument('--output_layer', default=[32,64]) 
  parser.add_argument('--num_samples', default=200) 
  parser.add_argument('--langevin_K', default=40)
  parser.add_argument('--langevin_s', default=0.5) 
  parser.add_argument('--kappa', default=10)
  parser.add_argument('--penalties', default=[10,20,50,100], type=int)
  parser.add_argument('--time_embed_dim', default=16, type=int)
  
  parser.add_argument('--epoch', default=50)
  parser.add_argument('--decoder_iteration', default=20)
  parser.add_argument('--nu_iteration', default=20)
  parser.add_argument('--decoder_lr', default=0.01)
  parser.add_argument('--decoder_thr', default=0.001)
  parser.add_argument('--iter_thr', default=5)
  parser.add_argument('--loglik_thr', default=0.00001)

  parser.add_argument('--use_data', default='XXX')
  parser.add_argument('--num_seq', default=10)
  parser.add_argument('--num_time', default=100)
  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('-f', required=False)

  return parser.parse_args()




args = parse_args(); print(args)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)


'''
# create folder
if args.use_data == 'XXX':
  output_dir = os.path.join()
os.makedirs(output_dir, exist_ok=True)
''' 


###################
# LOAD SAVED DATA #
###################


if args.use_data == 'XXX':
  print('[INFO]')
  data = torch.from_numpy(np.load(args.data_dir + 'XXX.npy'))

data = data.float()
print('[INFO] data loaded with dimension:', data.shape)



def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.05, 0.05)



#########
# MODEL #
#########

class CPD(nn.Module):
  def __init__(self, args, half):
    super(CPD, self).__init__()

    self.d = args.z_dim
    self.p = args.x_dim

    if half:
      self.T = int(args.num_time/2)
    else:
      self.T = args.num_time

    self.l1 = nn.Linear( self.d + self.p, args.output_layer[0])
    self.l2 = nn.Linear( args.output_layer[0] , args.output_layer[1])
    
    self.l3_pi    = nn.Linear(args.output_layer[1], args.K_dim)
    self.l3_mean  = nn.Linear(args.output_layer[1], args.K_dim * args.y_dim)
    self.l3_rho   = nn.Linear(args.output_layer[1], args.K_dim * args.y_dim)
    self.l3_sigma = nn.Linear(args.output_layer[1], args.K_dim * args.y_dim)


  def forward(self, x, z):
    # z: Tm by d

    output = torch.cat([x, z], dim=1)
    output = self.l1(output).tanh()
    output = self.l2(output).tanh()

    pi = self.l3_pi(output).softmax()  # K
    mean = self.l3_mean(output).reshape(-1, 3) # K by 3
    rho = self.l3_rho(output).tanh().reshape(-1, 3) # K by 3
    sigma = self.l3_sigma(output).relu().reshape(-1, 3) # K by 3

    # Create the covariance matrices
    covariance_matrices = []
    K = pi.shape[1]  # number of clusters (K)
    for k in range(K):
        sigma_k = sigma[k, :]  # 3 values for this cluster
        rho_k = rho[k, :]      # 3 values for this cluster
        
        # Construct the covariance matrix for cluster k (3x3)
        cov_matrix = torch.zeros(3, 3)
        cov_matrix[0, 0] = sigma_k[0] ** 2
        cov_matrix[1, 1] = sigma_k[1] ** 2
        cov_matrix[2, 2] = sigma_k[2] ** 2
        cov_matrix[0, 1] = cov_matrix[1, 0] = rho_k[0] * sigma_k[0] * sigma_k[1]
        cov_matrix[0, 2] = cov_matrix[2, 0] = rho_k[1] * sigma_k[0] * sigma_k[2]
        cov_matrix[1, 2] = cov_matrix[2, 1] = rho_k[2] * sigma_k[1] * sigma_k[2]

        covariance_matrices.append(cov_matrix)

    return pi, mean, rho, sigma, covariance_matrices
    


  def infer_z(self, x, z, y_repeat, mu_repeat):
    # z: Tm by d

    criterion = nn.MSELoss(reduction='sum')
    for k in range(args.langevin_K):
      z = z.detach().clone()
      z.requires_grad = True
      assert z.grad is None

      y_pred = self.forward(z)
      nll = criterion(y_pred, y_repeat)
      z_grad_nll = torch.autograd.grad(nll, z)[0]
      noise = torch.randn(self.T * args.num_samples, self.d).to(device) # Tm by d

      # Langevin dynamics sampling
      z = z + torch.tensor(args.langevin_s) * (-z_grad_nll - (z - mu_repeat)) +\
          torch.sqrt(2 * torch.tensor(args.langevin_s)) * noise
          
    z = z.detach().clone() # Tm by d
    return z

  def compute_covariance_matrix(self, rho, sigma):

        K = sigma.shape[0]  # Number of components (clusters)
        y_dim = sigma.shape[2]  # Dimensionality of the output (3 in your case)

        # Prepare an empty list to store covariance matrices for each component
        cov_matrices = []

        for k in range(K):
            # Extract the individual rho and sigma values for this component
            component_rho = rho[k]  # (3 choose 2) or 3 values
            component_sigma = sigma[k]  # 3 values (sigma_1, sigma_2, sigma_3)
            
            # Initialize a 3x3 covariance matrix
            cov_matrix = torch.zeros(y_dim, y_dim)

            # Set the diagonal elements (sigma_i^2)
            for i in range(y_dim):
                cov_matrix[i, i] = component_sigma[i] ** 2

            # Set the off-diagonal elements (rho_ij * sigma_i * sigma_j)
            off_diag_idx = 0
            for i in range(y_dim):
                for j in range(i + 1, y_dim):
                    rho_ij = component_rho[off_diag_idx]
                    cov_matrix[i, j] = rho_ij * component_sigma[i] * component_sigma[j]
                    cov_matrix[j, i] = cov_matrix[i, j]  # Symmetric covariance matrix
                    off_diag_idx += 1

            cov_matrices.append(cov_matrix)
        
        return torch.stack(cov_matrices, dim=0)  # Stack the covariance matrices along the batch dimension (K)


#################
# LOSS FUNCTION #   
#################


def gaussian_pdf(y, mean, cov_matrix):

    d = y.size(1)  # dimensionality of y
    cov_inv = torch.linalg.inv(cov_matrix)  # inverse of covariance matrix
    det_cov = torch.det(cov_matrix)  # determinant of covariance matrix
    
    diff = y - mean
    exponent = torch.sum(diff @ cov_inv * diff, dim=1)  # (y - mean)^T * Sigma_inv * (y - mean)
    
    # Multivariate Gaussian pdf
    coeff = 1 / ((2 * torch.pi) ** (d / 2) * torch.sqrt(det_cov))
    pdf_values = coeff * torch.exp(-0.5 * exponent)
    
    return pdf_values


def mixture_of_gaussians_loss(y, pi, mean, covariance_matrices):

    batch_size = y.size(0)
    K = pi.size(1)
    loss = 0.0
    
    for k in range(K):
        cov_k = covariance_matrices[k]  # Covariance matrix for cluster k (D x D)
        mean_k = mean[:, k, :]  # Mean for cluster k (B x D)
        
        # Calculate Gaussian pdf for each sample in the batch
        gaussian_values = gaussian_pdf(y, mean_k, cov_k)  # B (batch size)
        
        # Weight by the mixing coefficient pi_k
        weighted_gaussian_values = pi[:, k] * gaussian_values
        
        # Add to the total log-likelihood (negative sign for loss)
        loss -= torch.sum(torch.log(weighted_gaussian_values + 1e-10))  # Add small epsilon for numerical stability
    
    return loss / batch_size  # Average loss over the batch


############
# TRAINING #
############


def learn_one_seq_penalty(args, x_train, x_test, y_train, y_test, pen_iter, seq_iter, half):
  torch.manual_seed(1)
  
  m = args.num_samples
  kappa = args.kappa
  d = args.latent_dim
  penalty = args.penalties[pen_iter]
  early_stopping = False
  stopping_count = 0 # for ADMM

  if half:
    T = int(args.num_time/2)
    true_CP = args.true_CP_half
    label = 'half'
  else:
    T = args.num_time
    true_CP = args.true_CP_full
    label = 'full'

  # create matrix X and vector 1
  ones_col = torch.ones(T, 1).to(device)
  X = torch.zeros(T, T-1).to(device)
  i, j = torch.tril_indices(T, T-1, offset=-1)
  X[i, j] = 1 # Group Fused Lasso

  old_loglik = -float('inf')
  loglik_train_holder = []
  loglik_test_holder = []
  mu_diff_holder = []
  decoder_loss_holder = []
  CV_holder = []


  # use Coefficient of Variation (CV) when half=True
  # save result based on Coefficient of Variation (CV)
  if not half:
    best_mu = torch.zeros(T,d)
    best_loglik = torch.zeros(1)
    best_CV = -float('inf') # Coefficient of Variation
    best_CV_iter = 0

  
  # initialize mu, nu, w, with dim T by d of zeros
  mu = torch.zeros(T, d).to(device)
  nu = torch.zeros(T, d).to(device)
  w = torch.zeros(T, d).to(device)
  
  mu_old = mu.detach().clone()
  nu_old = nu.detach().clone()


  # creat repeated version of ground truth, from (T by n by n) to (Tm by n by n)
  # repeat m for T times, giving [m, m, ..., m]
  # for each t in T (axis=0), repeat num_samples times, giving (Tm by n by n)
  y_train_repeat = np.repeat(y_train.numpy(), np.repeat(m, T), axis=0) 
  y_train_repeat = torch.from_numpy(y_train_repeat).to(device)

  if half:
    # these objects exist if half = True
    y_test_repeat = np.repeat(y_test.numpy(), np.repeat(m, T), axis=0) 
    y_test_repeat = torch.from_numpy(y_test_repeat).to(device)


  model = CPD(args, half).to(device)
  model.apply(init_weights)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.decoder_lr)
  criterion = nn.MSELoss(reduction='sum') # loglik sum over t, but expectation over m, so later divided by m
  
  for learn_iter in range(args.epoch):

    ####################
    # GENERATE SAMPLES #
    ####################
    # create repeated version of mu, from (T by d) to (Tm by d)
    mu_repeat = np.repeat(mu.cpu().numpy(), np.repeat(m, T), axis=0)
    mu_repeat = torch.from_numpy(mu_repeat).to(device) # Tm by d
    init_z = torch.randn(T*m, d).to(device) # Tm by d, starts from N(0,1)
    sampled_z_all = model.infer_z(init_z, y_train_repeat, mu_repeat) # Tm by d

    ################
    # UPDATE PRIOR # 
    ################
    expected_z = sampled_z_all.clone().reshape(T,m,d) # T by m by d
    expected_z = expected_z.mean(dim=1) # T by d
    mu = ( expected_z + kappa * (nu-w) ) / ( 1.0 + kappa )
    mu = mu.detach().clone()

    ##################
    # UPDATE DECODER #
    ##################
    inner_loss = float('inf')
    for decoder_iter in range(args.decoder_iteration):
      optimizer.zero_grad()
      y_pred = model(sampled_z_all) # Tm by n by n 
      loss = criterion(y_pred, y_train_repeat) / m
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()

    #########################
    # UPDATE BETA AND GAMMA #
    #########################

    gamma = nu[0, :].unsqueeze(0) # row vector
    beta = torch.diff(nu, dim=0)

    for nu_iter in range(args.nu_iteration):
      # update beta once (t range from 1 to 99, NOT 1 to 100)
      for t in range(T-1): 
        beta_without_t = beta.detach().clone()
        X_without_t = X.detach().clone()
        beta_without_t[t,:] = torch.zeros(d) # make this row zeros
        X_without_t[:,t] = torch.zeros(T) # make this column zeros
        bt = kappa * torch.mm( X[:,t].unsqueeze(0), mu + w - torch.mm(ones_col, gamma) - torch.mm(X_without_t, beta_without_t) )
        bt_norm = torch.norm(bt, p=2)

        # UPDATE: soft-thresholding
        if bt_norm < penalty:
          beta[t,:] = torch.zeros(d)
        else:
          beta[t,:] = 1 / (kappa * torch.norm(X[:,t], p=2)**2) * (1 - penalty/bt_norm) * bt
        beta = beta.detach().clone()

      # update gamma
      gamma = torch.mean(mu + w - torch.mm(X, beta), dim=0).unsqueeze(0).detach().clone()

    # recollect nu
    nu = torch.mm(ones_col, gamma) + torch.mm(X, beta)
    nu = nu.detach().clone()

    ############
    # UPDATE W # 
    ############

    w = mu - nu + w
    w = w.detach().clone()

    ############
    # RESIDUAL # 
    ############

    primal_residual = torch.sqrt(torch.mean(torch.square(mu - nu)))
    dual_residual = torch.sqrt(torch.mean(torch.square(nu - nu_old)))

    if primal_residual > 10.0 * dual_residual:
      kappa *= 2.0
      w *= 0.5
      print('\n[INFO] kappa increased to', kappa)
    elif dual_residual > 10.0 * primal_residual:
      kappa *= 0.5
      w *= 2.0
      print('\n[INFO] kappa decreased to', kappa)

    
    # calculate log_likelihood
    with torch.no_grad():
      '''
      # if half=False, adj_test_repeat does not exist
      if half:
        loglik_train = model.cal_loglik(mu, y_train_repeat) # USE TRAIN
        loglik_train_holder.append(loglik_train.detach().cpu().numpy().item())
        loglik = model.cal_loglik(mu, adj_test_repeat) # USE TEST
        loglik_test_holder.append(loglik.detach().cpu().numpy().item())
      else:
        loglik = model.cal_loglik(mu, adj_train_repeat) # USE TRAIN
        loglik_train_holder.append(loglik.detach().cpu().numpy().item())
      
      # criteria 1
      loglik_relative_diff = torch.abs((loglik - old_loglik) / old_loglik)
      old_loglik = loglik.detach().clone()
      '''

      # criteria 2
      mu_relative_diff = torch.norm(mu-mu_old, p='fro')
      mu_diff_holder.append(mu_relative_diff.detach().cpu().numpy().item())
      
      mu_old = mu.detach().clone()
      nu_old = nu.detach().clone()

    #####################
    # STOPPING CRITERIA #
    #####################
 
    if loglik_relative_diff < args.loglik_thr:
      stopping_count += 1
    else:
      stopping_count = 0

    if stopping_count >= args.iter_thr:
      print('\n[INFO] early stopping')
      early_stopping = True



    ##############
    # PRINT INFO #
    ##############

    if (learn_iter+1) % 10 == 0:

      with torch.no_grad():
        # second row - first row
        delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1)
        delta_mu = delta_mu.cpu().detach().numpy() # numpy for plot

        plt.plot(delta_mu); plt.xticks(true_CP)
        plt.savefig( output_dir + '/{}_delta_mu_seq{}pen{}_learn{}'.format(label,seq_iter,pen_iter,learn_iter+1) + '.png' ) 
        plt.close()

        if half: 
          plt.plot(loglik_train_holder[1:], label="train")
          plt.plot(loglik_test_holder[1:], label="test")
          plt.legend(loc="lower right")
          plt.savefig( output_dir + '/{}_loglik_seq{}pen{}'.format(label,seq_iter,pen_iter) + '.png' ) 
          plt.close()
        else:
          plt.plot(loglik_train_holder[1:])
          plt.savefig( output_dir + '/{}_loglik_seq{}pen{}'.format(label,seq_iter,pen_iter) + '.png' ) 
          plt.close()

        plt.plot(mu_diff_holder[1:])
        plt.savefig( output_dir + '/{}_mu_diff_seq{}pen{}'.format(label,seq_iter,pen_iter) + '.png' ) 
        plt.close()

        if not half:
          plt.plot(CV_holder)
          plt.savefig( output_dir + '/{}_CV_seq{}pen{}'.format(label,seq_iter,pen_iter) + '.png' ) 
          plt.close()

        print('\nlearning iter (seq={}, [penalty={}], data={}) ='.format(seq_iter,penalty,label), learn_iter+1, 'of', args.epoch)
        print('\tlog likelihood =', loglik)
        print('\tprimal residual =', primal_residual)
        print('\tdual residual =', dual_residual)
        print('\t\tlog likelihood relative difference =', loglik_relative_diff)
        print('\t\tmu relative difference =', mu_relative_diff)
      
    ###############
    # SAVE RESULT #
    ###############
    # at the last iteration or early_stopping
    if (learn_iter+1) == args.epoch or early_stopping:
      print('\nFINAL learning iter (seq num = {}, [penalty={}]) ='.format(seq_iter,penalty), learn_iter+1, 'of', args.epoch)
      print('FINAL log likelihood =', loglik)
      print('FINAL log likelihood relative difference =', loglik_relative_diff)
      print('FINAL mu relative difference =', mu_relative_diff)

      with torch.no_grad():
        if half:
          # USE THE LAST MU
          delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1)
          result = evaluation(delta_mu, args, loglik, pen_iter, seq_iter, output_dir, half)
          return torch.tensor(result[0:5]), loglik, mu
        else:
          # USE THE BEST MU
          print('[INFO] best_CV_iter =', best_CV_iter)
          delta_mu = torch.norm(torch.diff(best_mu, dim=0), p=2, dim=1)
          result = evaluation(delta_mu, args, best_loglik, pen_iter, seq_iter, output_dir, half)
          return torch.tensor(result[0:5]), best_loglik, best_mu, w_left, w_right

      

######################
# parameter learning #
######################


# results holder
output_holder = torch.zeros(args.num_seq, 5) # (4 metrics + loglik) for all sequences


for seq_iter in range(0,args.num_seq):

  #if seq_iter == 1: break # test code on one sequence

  print('\n[INFO] sequence =', seq_iter)
  best_loglik = torch.tensor(-float('inf')) # keep updated throughout cross-validation
  #output_seq_holder = torch.zeros(len(args.penalties), 5) # (4 metrics + loglik) for all penalties of one sequence


  # visualization of edge count
  sums = one_seq.reshape(args.num_time, -1).sum(dim=1)
  plt.plot(list(range(1, args.num_time + 1)), sums)
  plt.axvline(x=26,color='r'); plt.axvline(x=51,color='r'); plt.axvline(x=76,color='r')
  plt.xticks(args.true_CP_full); plt.title('Edge Count over Time')
  plt.savefig(output_dir + '/edge_count_seq{}.png'.format(seq_iter))
  plt.close()
  

  ##############
  # SPLIT DATA #
  ##############
  odd_idx = range(1, args.num_time, 2)
  even_idx = range(0, args.num_time, 2)
  train_data = one_seq[odd_idx,:,:]
  test_data = one_seq[even_idx,:,:]
  print('[INFO] train_data.shape:', train_data.shape)
  print('[INFO] test_data.shape:', test_data.shape)

  ######################
  # TRAIN ON HALF DATA #
  ######################
  holder_loglik = []
  holder_index_comb = []

  # same sequence, different penalties
  for pen_iter in range(len(args.penalties)):
    test_result, test_loglik, _ =\
    learn_one_seq_penalty(args, train_data, test_data, pen_iter, seq_iter, half=True)
  
    holder_loglik.append(test_loglik.cpu().numpy().item())
    holder_index_comb.append(pen_iter)
    print('test_result:\n',test_result)
    print('holder_loglik:\n',holder_loglik)
    print('holder_index_comb:\n',holder_index_comb)

    if test_loglik > best_loglik:
      print('[INFO] best_loglik is updated')
      best_loglik = test_loglik

  # model selection via Cross Validation
  best_comb_index = holder_loglik.index(max(holder_loglik))
  best_comb = holder_index_comb[best_comb_index] # return a pen_iter correspond to highest test_loglik
  
  pen_selection = best_comb
  print('[INFO] max test loglik at pen_iter={}'.format(pen_selection))

  #####################
  # TEST ON FULL DATA #
  #####################

  # use one_seq and pen_iter_selection with 'test_data = None'
  output_holder[seq_iter,:], _, mu_seq_output, w_left_seq_output, w_right_seq_output = \
  learn_one_seq_penalty(args, one_seq, None, pen_selection, seq_iter, half=False) 
  print('[INFO] result for seq_iter = {}:\n'.format(seq_iter), output_holder[seq_iter,:])
  torch.save(output_holder, os.path.join(output_dir, 'result_table_seq{}.pt'.format(seq_iter)) ) # have previous results
  torch.save(mu_seq_output, os.path.join(output_dir, 'mu_par_seq{}.pt'.format(seq_iter)) ) # best mu for this seq_iter
  np.save(os.path.join(output_dir,'w_left_seq{}.npy'.format(seq_iter)), w_left_seq_output.detach().cpu().numpy())
  np.save(os.path.join(output_dir,'w_right_seq{}.npy'.format(seq_iter)), w_right_seq_output.detach().cpu().numpy())



# save final result for all sequences
print('output_holder:\n', output_holder)
torch.save(output_holder, os.path.join(output_dir, 'result_table_all.pt') ) # save all results in the last

# print result for table
print('mean perforamnce:\n', np.mean(output_holder.cpu().numpy(),axis=0)[0:4])
print('std perforamnce:\n', np.std(output_holder.cpu().numpy(),axis=0)[0:4])














