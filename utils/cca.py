import torch
import numpy as np

num_cca_trials = 5

def positivedef_matrix_sqrt(array):
  """Stable method for computing matrix square roots, supports complex matrices.
  Args:
            array: A numpy 2d array, can be complex valued that is a positive
                   definite symmetric (or hermitian) matrix
  Returns:
            sqrtarray: The matrix square root of array
  """
  w, v = torch.linalg.eigh(array)
  #  A - torch.matmul(v, torch.matmul(torch.diag(w), v.T))
  wsqrt = torch.sqrt(w).nan_to_num()
  sqrtarray = torch.matmul(v, torch.matmul(torch.diag(wsqrt), torch.conj(v).T))
  return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
  """Takes covariance between X, Y, and removes values of small magnitude.
  Args:
            sigma_xx: 2d numpy array, variance matrix for x
            sigma_xy: 2d numpy array, crossvariance matrix for x,y
            sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                      (conjugate) transpose of sigma_xy
            sigma_yy: 2d numpy array, variance matrix for y
            epsilon : cutoff value for norm below which directions are thrown
                       away
  Returns:
            sigma_xx_crop: 2d array with low x norm directions removed
            sigma_xy_crop: 2d array with low x and y norm directions removed
            sigma_yx_crop: 2d array with low x and y norm directiosn removed
            sigma_yy_crop: 2d array with low y norm directions removed
            x_idxs: indexes of sigma_xx that were removed
            y_idxs: indexes of sigma_yy that were removed
  """

  x_diag = torch.abs(torch.diagonal(sigma_xx))
  y_diag = torch.abs(torch.diagonal(sigma_yy))
  x_idxs = (x_diag >= epsilon)
  y_idxs = (y_diag >= epsilon)

  sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
  sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
  sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
  sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

  '''print("\n sigx", sigma_xx_crop)
  print("sigxy", sigma_xy_crop)
  print("sigyx", sigma_yx_crop)
  print("sigy", sigma_yy_crop)'''

  return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,
          x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon,
                 verbose=True):
  """Main cca computation function, takes in variances and crossvariances.
  This function takes in the covariances and cross covariances of X, Y,
  preprocesses them (removing small magnitudes) and outputs the raw results of
  the cca computation, including cca directions in a rotated space, and the
  cca correlation coefficient values.
  Args:
            sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                      variance matrix for x
            sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                      crossvariance matrix for x,y
            sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                      crossvariance matrix for x,y (conj) transpose of sigma_xy
            sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                      variance matrix for y
            epsilon:  small float to help with stabilizing computations
            verbose:  boolean on whether to print intermediate outputs
  Returns:
            [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                          ux and vx are (conj) transposes of each other, being
                          the canonical directions in the X subspace.
                          sx is the set of canonical correlation coefficients-
                          how well corresponding directions in vx, Vy correlate
                          with each other.
            [uy, sy, vy]: Same as above, but for Y space
            invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                          directions back to original space
            invsqrt_yy:   Same as above but for sigma_yy
            x_idxs:       The indexes of the itorchut sigma_xx that were pruned
                          by remove_small
            y_idxs:       Same as above but for sigma_yy
  """

  (sigma_xx, sigma_xy, sigma_yx, sigma_yy,
   x_idxs, y_idxs) = remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)

  numx = sigma_xx.shape[0]
  numy = sigma_yy.shape[0]

  if numx == 0 or numy == 0:
    return ([0, 0, 0], torch.zeros_like(sigma_xx),
            torch.zeros_like(sigma_yy), x_idxs, y_idxs)

  if verbose:
    print("adding eps to diagonal and taking inverse")
  sigma_xx += epsilon * torch.eye(numx, device=sigma_xx.device)
  sigma_yy += epsilon * torch.eye(numy, device=sigma_xx.device)
  inv_xx = torch.linalg.pinv(sigma_xx)
  inv_yy = torch.linalg.pinv(sigma_yy)

  if verbose:
    print("taking square root")
  invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
  invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

  if verbose:
    print("dot products...")
  arr = torch.matmul(invsqrt_xx, torch.matmul(sigma_xy, invsqrt_yy))

  '''print()
  print(invsqrt_xx)
  print(invsqrt_yy)
  print(arr)'''

  if verbose:
    print("trying to take final svd")
  u, s, v = torch.linalg.svd(arr)

  if verbose:
    print("computed everything!")

  return [u, torch.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
  """Computes threshold index of decreasing nonnegative array by summing.
  This function takes in a decreasing array nonnegative floats, and a
  threshold between 0 and 1. It returns the index i at which the sum of the
  array up to i is threshold*total mass of the array.
  Args:
            array: a 1d numpy array of decreasing, nonnegative floats
            threshold: a number between 0 and 1
  Returns:
            i: index at which torch.sum(array[:i]) >= threshold
  """
  assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

  for i in range(len(array)):
    if torch.sum(array[:i])/torch.sum(array) >= threshold:
      return i


def create_zero_dict(compute_dirns, dimension):
  """Outputs a zero dict when neuron activation norms too small.
  This function creates a return_dict with appropriately shaped zero entries
  when all neuron activations are very small.
  Args:
            compute_dirns: boolean, whether to have zero vectors for directions
            dimension: int, defines shape of directions
  Returns:
            return_dict: a dict of appropriately shaped zero entries
  """
  return_dict = {}
  return_dict["mean"] = (torch.asarray(0), torch.asarray(0))
  return_dict["sum"] = (torch.asarray(0), torch.asarray(0))
  return_dict["cca_coef1"] = torch.asarray(0)
  return_dict["cca_coef2"] = torch.asarray(0)
  return_dict["idx1"] = 0
  return_dict["idx2"] = 0

  if compute_dirns:
    return_dict["cca_dirns1"] = torch.zeros((1, dimension))
    return_dict["cca_dirns2"] = torch.zeros((1, dimension))

  return return_dict


def get_cca_similarity(acts1, acts2, epsilon=0., threshold=0.98,
                       compute_coefs=True,
                       compute_dirns=False,
                       verbose=False):
  """The main function for computing cca similarities.
  This function computes the cca similarity between two sets of activations,
  returning a dict with the cca coefficients, a few statistics of the cca
  coefficients, and (optionally) the actual directions.
  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints
            epsilon: small float to help stabilize computations
            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.
            compute_coefs: boolean value determining whether coefficients
                           over neurons are computed. Needed for computing
                           directions
            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)
            verbose: Boolean, whether intermediate outputs are printed
  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  if not acts1.shape[0] < acts1.shape[1]:
    print(acts1.shape)
  # assert dimensionality equal
  assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
  # check that acts1, acts2 are transposition
  assert acts1.shape[0] < acts1.shape[1], ("itorchut must be number of neurons"
                                           "by datapoints")
  return_dict = {}

  # compute covariance with numpy function for extra stability
  numx = acts1.shape[0]
  numy = acts2.shape[0]

  acts12 = torch.concat((acts1, acts2), axis=0)
  covariance = torch.cov(acts12)
  sigmaxx = covariance[:numx, :numx]
  sigmaxy = covariance[:numx, numx:]
  sigmayx = covariance[numx:, :numx]
  sigmayy = covariance[numx:, numx:]

  # rescale covariance to make cca computation more stable
  xmax = torch.max(torch.abs(sigmaxx))
  ymax = torch.max(torch.abs(sigmayy))
  sigmaxx /= xmax
  sigmayy /= ymax
  sigmaxy /= torch.sqrt(xmax * ymax)
  sigmayx /= torch.sqrt(xmax * ymax)

  ([u, s, v], invsqrt_xx, invsqrt_yy,
   x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                                  epsilon=epsilon,
                                  verbose=verbose)

  # if x_idxs or y_idxs is all false, return_dict has zero entries
  if (not torch.any(x_idxs)) or (not torch.any(y_idxs)):
    return create_zero_dict(compute_dirns, acts1.shape[1])

  if compute_coefs:
    
    # also compute full coefficients over all neurons
    np_x_idxs, np_y_idxs = x_idxs.detach().to("cpu").numpy(), y_idxs.detach().to("cpu").numpy()
    #x_mask = torch.from_numpy(np.dot(np_x_idxs.reshape((-1, 1)), np_x_idxs.reshape((1, -1)))).to(x_idxs.device)
    #y_mask = torch.from_numpy(np.dot(np_y_idxs.reshape((-1, 1)), np_y_idxs.reshape((1, -1)))).to(x_idxs.device)

    x_mask = np.dot(np_x_idxs.reshape((-1, 1)), np_x_idxs.reshape((1, -1)))
    y_mask = np.dot(np_y_idxs.reshape((-1, 1)), np_y_idxs.reshape((1, -1)))

    return_dict["coef_x"] = u.T
    return_dict["invsqrt_xx"] = invsqrt_xx

    return_dict["full_coef_x"] = np.zeros((numx, numx))
    np.place(return_dict["full_coef_x"], x_mask,
             return_dict["coef_x"].detach().to("cpu").numpy())
    return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
    np.place(return_dict["full_invsqrt_xx"], x_mask,
             return_dict["invsqrt_xx"].detach().to("cpu").numpy())

    return_dict["full_coef_x"] = torch.from_numpy(return_dict["full_coef_x"]).to(invsqrt_xx.device).float()
    return_dict["full_invsqrt_xx"] = torch.from_numpy(return_dict["full_invsqrt_xx"]).to(invsqrt_xx.device).float()
    
    return_dict["coef_y"] = v
    return_dict["invsqrt_yy"] = invsqrt_yy

    return_dict["full_coef_y"] = np.zeros((numy, numy))
    np.place(return_dict["full_coef_y"], y_mask,
             return_dict["coef_y"].detach().to("cpu").numpy())
    return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
    np.place(return_dict["full_invsqrt_yy"], y_mask,
             return_dict["invsqrt_yy"].detach().to("cpu").numpy())

    return_dict["full_coef_y"] = torch.from_numpy(return_dict["full_coef_y"]).to(invsqrt_xx.device).float()
    return_dict["full_invsqrt_yy"] = torch.from_numpy(return_dict["full_invsqrt_yy"]).to(invsqrt_xx.device).float()

    '''
    return_dict["full_coef_x"] = torch.zeros((numx, numx))
    return_dict["full_invsqrt_xx"] = torch.zeros((numx, numx))

    return_dict["full_coef_x"] = return_dict["coef_x"] * x_mask.int().float()
    return_dict["full_invsqrt_xx"] = return_dict["invsqrt_xx"] * x_mask.int().float()

    return_dict["full_coef_y"] = torch.zeros((numy, numy))
    return_dict["full_invsqrt_yy"] = torch.zeros((numy, numy))

    return_dict["full_coef_y"] = return_dict["coef_y"] * y_mask.int().float()
    return_dict["full_invsqrt_yy"] = return_dict["invsqrt_yy"] * y_mask.int().float()

    '''

    # compute means
    neuron_means1 = torch.mean(acts1, axis=1, keepdims=True)
    neuron_means2 = torch.mean(acts2, axis=1, keepdims=True)
    return_dict["neuron_means1"] = neuron_means1
    return_dict["neuron_means2"] = neuron_means2

  if compute_dirns:
    # orthonormal directions that are CCA directions
    cca_dirns1 = torch.matmul(torch.matmul(return_dict["full_coef_x"],
                               return_dict["full_invsqrt_xx"]),
                        (acts1 - neuron_means1)) + neuron_means1
    cca_dirns2 = torch.matmul(torch.matmul(return_dict["full_coef_y"],
                               return_dict["full_invsqrt_yy"]),
                        (acts2 - neuron_means2)) + neuron_means2

  # get rid of trailing zeros in the cca coefficients
  idx1 = sum_threshold(s, threshold)
  idx2 = sum_threshold(s, threshold)

  return_dict["cca_coef1"] = s
  return_dict["cca_coef2"] = s
  return_dict["x_idxs"] = x_idxs
  return_dict["y_idxs"] = y_idxs
  # summary statistics
  return_dict["mean"] = (torch.mean(s[:idx1]), torch.mean(s[:idx2]))
  return_dict["sum"] = (torch.sum(s), torch.sum(s))

  if compute_dirns:
    return_dict["cca_dirns1"] = cca_dirns1
    return_dict["cca_dirns2"] = cca_dirns2

  return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-5,
                          compute_dirns=False):
  """Calls get_cca_similarity multiple times while adding noise.
  This function is very similar to get_cca_similarity, and can be used if
  get_cca_similarity doesn't converge for some pair of itorchuts. This function
  adds some noise to the activations to help convergence.
  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints
            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.
            epsilon: small float to help stabilize computations
            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)
  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  for trial in range(num_cca_trials):
    try:
      return_dict = get_cca_similarity(acts1, acts2, epsilon=epsilon, threshold=threshold, compute_dirns=compute_dirns)
    except:
      acts1 = acts1*1e-1 + torch.normal(mean=0.0, std=1.0, size=acts1.shape).to(acts1.device)*epsilon
      acts2 = acts2*1e-1 + torch.normal(mean=0.0, std=1.0, size=acts2.shape).to(acts1.device)*epsilon
      if trial + 1 == num_cca_trials:
        raise

  return return_dict