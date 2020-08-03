function [diagKtotal, diagKi]= gaussProc_evalKernelDiag(X1, opt)
% gaussProc_evalKernelDiag - Evaluate kernel diagonal elements, taking feature indexing and kernel weights into account
%
% Synopsis:
%   K = gaussProc_evalKernelDiag(X1,opt)
%   
% Arguments:
%   X1: [d N1] matrix. Input data for evaluating kernel (N1 points in d
%       dimensions)
%   opt: Options structure for GP model. Required fields are
%       'kernel' (name(s) and params of all kernel functions)
%       'kernelweight' (scaling for each kernel function)
%       'kernelindex' (feature indices for each kernel)
%   
% Returns:
%   diagK: [1 N1] vector. Vector of pairwise kernel evaluations, with
%      diagK(i) being the result of evaluating the kernel function on the
%      pair of points k(X1(:,i), X1(:,i))
%   
% Description:
%   This routine returns the "diagonal elements" of a kernel matrix, if
%   possible without excessive computational overhead (i.e., loops).
%   Computing the diagonal elements is easy for stationary kernels (such
%   as rbf and ratquad), but more tricky for neural network kernels.
%   Currently, the distinction between stationary and nonstationary
%   kernels is only done by a simple lookup list. If you implement new
%   kernel functions, this lookup list needs to be modified.
%
%   The kernel function that is effectively used is
%     sum_{i=1}^k exp(opt.kernelweight(i)) * k_i(x1,x2)
%   respectively with feature indices, in this case the individual
%   contribution is
%     k_i(x1(opt.kernelindex{i}),x1(opt.kernelindex{i}))
%
% See also: 
%

% Author(s): Anton Schwaighofer, Joaquin Quiñonero-Candela, Feb 2007
% $Id: gaussProc_evalKernelDiag.m,v 1.2 2007/02/20 13:11:40 neuro_toolbox Exp $


error(nargchk(2, 2, nargin));

nKernels = length(opt.kernel);
if nargout>1,
  Ki = cell([1 nKernels]);
end
diagKtotal = [];
for i = 1:nKernels,
  if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
    % No index given at all or nothing given for this particular
    % kernel: assume they operate on all data
    diagK = evalKernelDiag(X1, opt.kernel{i});
  else
    % Index given:
    diagK = evalKernelDiag(X1(opt.kernelindex{i},:), opt.kernel{i});
  end
  % Build the linear combination of kernels for multi-kernel learning
  if isempty(diagKtotal),
    diagKtotal = exp(opt.kernelweight(i))*diagK;
  else
    diagKtotal = diagKtotal + exp(opt.kernelweight(i))*diagK;
  end
  % Store individual kernel matrices only on request
  if nargout>1,
    diagKi{i} = diagK;
  end
end


function diagK = evalKernelDiag(X1,kernel)
% evalKernelDiag - Helper function to evaluate the diagonal of a kernel function on a set of points
%
% Synopsis:
%   K = evalKernel(X1,kernel)
%   
% Arguments:
%   X1: [d N1] matrix. Input data for evaluating kernel (N1 points in d
%       dimensions)
%   kernel: String or cell array. If string, the name of the kernel function to
%       eval is 'kern_' plus the given string. If a cell, the kernel function name
%       is 'kern_' plus the content of kernel{1}, all further entries in kernel will
%       be passed as parameters to the kernel function.
%   
% Returns:
%   diagK: [1 N1] vector. Vector of pairwise kernel evaluations, with
%      diagK(i) being the result of evaluating the kernel function on the
%      pair of points k(X1(:,i), X1(:,i))
%   

error(nargchk(2, 2, nargin));

[dim, N] = size(X1);

[func0, param] = getFuncParam(kernel);
func = ['kern_' func0];

% if the kernel function has only one output parameter then assume the
% kernel is stationary
switch func0
  case {'rbf', 'ratquad', 'rbfratquad', 'const', 'matern'}
    % These are all stationary kernels
    % Simply replicate one kernel evaluation
    diagK = feval(func, X1(:,1), [], param{:}) * ones(1,N);
  case 'neuralnet'
    % neural net kernel has two output args, designed to return the
    % diagonal elements
    [dummy diagK] = feval(func, X1, [], param{:});
  otherwise
    % Don't know what to do... a kernel we have not seen yet. Let's be
    % optimistic and think it's stationary, but issue a warning
    warning('gaussProc_evalKernelDiag: Using an unknown kernel function');
    fprintf('Don''t know whether kernel function %s is stationary.\n', func);
    fprintf('Update gaussProc_evalKernelDiag accordingly.\n');
    fprintf('I will for now assume that the kernel is stationary.\n');
    diagK = feval(func, X1(:,1), [], param{:}) * ones(1,N);
end
