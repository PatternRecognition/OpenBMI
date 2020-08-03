function [Ktotal, Ki] = gaussProc_evalKernel(X1, X2, opt)
% gaussProc_evalKernel - Evaluate kernel function, taking feature indexing and kernel weights into account
%
% Synopsis:
%   K = gaussProc_evalKernel(X1,X2,opt)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for evaluating kernel (N1 points in d
%       dimensions)
%   X2: [d N2] matrix. Input data 2 for evaluating kernel (N2 points in d
%       dimensions). If left empty, X2==X1 will be assumed
%   opt: Options structure for GP model. Required fields are
%       'kernel' (name(s) and params of all kernel functions)
%       'kernelweight' (scaling for each kernel function)
%       'kernelindex' (feature indices for each kernel)
%   
% Returns:
%   K: [N1 N2] matrix. Pairwise kernel evaluations, with K(i,j) being the result
%       of evaluating the kernel function on the pair of points X1(:,i) and X2(:,j)
%   
% Description:
%   This routine evaluates a (set of) kernel functions on a set of
%   points. The kernel function that is effectively used is
%     sum_{i=1}^k exp(opt.kernelweight(i)) * k_i(x1,x2)
%   respectively with feature indices, in this case the individual
%   contribution is
%     k_i(x1(opt.kernelindex{i}),x2(opt.kernelindex{i}))
%   
%   
% See also: evalKernel
%

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: gaussProc_evalKernel.m,v 1.3 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(3, 3, nargin));

nKernels = length(opt.kernel);
if nargout>1,
  Ki = cell([1 nKernels]);
end
Ktotal = [];
for i = 1:nKernels,
  if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
    % No index given at all or nothing given for this particular
    % kernel: assume they operate on all data
    K = evalKernel(X1, X2, opt.kernel{i});
  else
    % Index given:
    if isempty(X2), 
      K = evalKernel(X1(opt.kernelindex{i},:), [], opt.kernel{i});
    else
      K = evalKernel(X1(opt.kernelindex{i},:), X2(opt.kernelindex{i},:), opt.kernel{i});
    end
  end
  % Build the linear combination of kernels for multi-kernel learning
  if isempty(Ktotal),
    Ktotal = exp(opt.kernelweight(i))*K;
  else
    Ktotal = Ktotal + exp(opt.kernelweight(i))*K;
  end
  % Store individual kernel matrices only on request
  if nargout>1,
    Ki{i} = K;
  end
end
