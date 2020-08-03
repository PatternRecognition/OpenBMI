function [invA, logdetA, q] = inv_logdet_pd(A)
% inv_logdet_pd - Compute inverse and log determinant of a positive definite matrix
%
% Synopsis:
%   invA = inv_logdet_pd(A)
%   [invA, logdetA, q] = inv_logdet_pd(A)
%   
% Arguments:
%  A: [n n] matrix. A must be positive definite, an error message is
%      issued if it is not
%   
% Returns:
%  invA: [n n] matrix. Inverse of A
%  logdetA: Scalar, this is log(det(A))
%  q: Scalar. If the input A is a positive matrix, q==0 upon return. If
%     q>0, the Cholesky decomposition could not be computed. A and
%     logdetA are invalid in this case. If the third output arg is not
%     provided, the routine will issue an error message if A is not
%     positive definite.
%   
% Description:
%   This function computes the inverse of a square, positive definite
%   symmetric matrix, and (optionally) the log of its determinant. This
%   function is about twice as fast as the builtin "inv" function, and
%   can compute inverse and log(det(A)) at virtually the same cost as the
%   inversion.
%   The routine is based on the Cholesky decomposition, and is thus
%   numerically relatively stable.
%   
% Examples:
%   To compute the inverse of a kernel matrix K:
%     K = K+eps^(1/2)*speye(size(K,1));
%     invK = inv_logdet_pd(K);
%
%   Adding a bit of jitter along the diagonal is recommended to improve
%   stability. Cholesky may refuse to work if the matrix is nearly
%   singular.
%   
% See also: inv,chol
% 

% Documentation to the routine inv_logdet_pd by Carl Rasmussen,
% Anton Schwaighofer, Aug 2005
% $Id: inv_logdet_pd.m,v 1.2 2006/06/19 19:59:14 neuro_toolbox Exp $

