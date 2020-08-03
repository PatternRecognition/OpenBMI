function K = evalKernel(X1,X2,kernel)
% evalKernel - Helper function to evaluate a kernel function on a set of points
%
% Synopsis:
%   K = evalKernel(X1,X2,kernel)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for evaluating kernel (N1 points in d
%       dimensions)
%   X2: [d N2] matrix. Input data 2 for evaluating kernel (N2 points in d
%       dimensions). If left empty, X2==X1 will be assumed
%   kernel: String or cell array. If string, the name of the kernel function to
%       eval is 'kern_' plus the given string. If a cell, the kernel function name
%       is 'kern_' plus the content of kernel{1}, all further entries in kernel will
%       be passed as parameters to the kernel function.
%   
% Returns:
%   K: [N1 N2] matrix. Pairwise kernel evaluations, with K(i,j) being the result
%       of evaluating the kernel function on the pair of points X1(:,i) and X2(:,j)
%   
% Description:
%   This routine evaluates a kernel function on a set of points. The
%   kernel function is here given by its name and eventual parameters,
%   that are passed on to the kernel function.
%   
%   
% Examples:
%   X1 = [2 2; 3 4]'; X2 = [2 2; 2 3; 4 4]';
%   evalKernel(X1, X2, 'rbf')
%     evaluates the RBF kernel with its default parameter settings on the
%     given sets of points. Result is a kernel matrix of size [2 3]
%   evalKernel(X1, X2, {'rbf', 'inweights', [0.5 1], 'bias', -Inf})
%     evaluates the RBF kernel with parameters 'inweights' and 'bias'
%   
% See also: kern_rbf,getFuncParam
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: evalKernel.m,v 1.2 2005/09/02 14:53:15 neuro_toolbox Exp $

error(nargchk(3, 3, nargin));

[func, param] = getFuncParam(kernel);
func = ['kern_' func];
K = feval(func, X1, X2, param{:});
