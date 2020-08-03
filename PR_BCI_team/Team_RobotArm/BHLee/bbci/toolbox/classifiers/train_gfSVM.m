function gfSVM = train_gfSVM(data, labels, varargin);
% train_gfSVM - trains a Suport Vector Machine (SVM) using the genefinder program
%
% Synopsis:
%   gfSVM = train_gfSVM(data, labels, properties)
%   gfSVM = train_gfSVM(data, labels, 'Property', Value, ...)
%   
% Arguments:
%   data:	[d,n]	real valued training data containing n examples of d dims
%	labels:	[2,n]	logical labels (2-class problems only!) or [1 d]
%                       vector with +1/-1 entries
%
% Returns:
%   gfSVM:	a trained SVM with fields:
%             'SV':     the support vectors
%             'alphas': the alphas for each support vector
%             'b':      the bias
%             'kernel': the kernel with parameters as string
%
% Properties:
%	'C': scalar or [1 2] matrix. Regularization parameter C
%                       If C is a [1 2] matrix, class wise regularization
%                       will be used, with C(1) the parameter for class 1
%                       (the "-1 class") and C(2) for class 2 (the "+1 class")
%
%       'C_weight'      weight by which the regularization parameter C will 
%                       be multiplied for the smaller class (default 1). 
%                       Only used when length(C) == 1
%
% 	'kernel': the name of the kernel to be used
%			linear: K(u,v) = u'*v;
%			poly: K(u,v) = (u'*v)^d when homogeneous
%				  K(u,v) = (u'*v+1)^d when inhomogeneous
%			gaussian: K(u,v) =  exp(-||u-v||^2/width)
%
%	'implementation': Which SVM implementation to use:
%			light: SVMLight (see http://svmlight.joachims.org) (default)
%			libsvm: libSVM (see http://www.csie.ntu.edu.tw/~cjlin/libsvm )
%
%       'verbosity'     0, 1 or 2. 
%                       If 0: display errors only.
%                       If 1: display errors and warnings (Default)
%                       If 2: debug output
%	'width':	gaussian kernel width
%	'degree:'	degree of the polynomial
%	'epsilon':  optimization accuracy (default 1e-5)
%	'cachesize': size of the kernel cache in megabytes (default 100MB)
%	'poly_inhomogene:'	if set to 0 homogene polynomial kernel, set to 1 for
%						inhomogene kernel
%
% Description:
%   The function train_SVM trains a Support Vector Machine, i.e. it
%	determines the support vectors x_i, lagrange multipliers alpha_i and
%	the bias term b of the following function
%		f(x)=sum_{i=0}^l alpha_i K(x_i,x) + b.
%	The training algorithm maximizes the margin between the classes and
%	solves a quadratic program (see References for more details).
%
% Examples:
%   x = [randn(10,50)-0.5, randn(10,50)];
%	y = [repmat([1;0], 1, 50) repmat([0;1], 1,50)];
%
%
%   s = train_gfSVM(x,y, 'C',1, 'kernel','linear', 'implementation','light');
%			trains a standard linear SVM using SVMlight. not very useful.
%
%   s = train_gfSVM(x,y, 'C',5, 'kernel','poly', 'degree', 3, 'svm','libsvm');
%			trains an inhomogenious polynomial svm of degree 3 using LibSVM
%
%	c=train_gfSVM(x, y, 'width', 10, 'cachesize', 100, 'C', [1 2] , 'kernel', 'gaussian', 'implementation', 'light');
%	out=apply_gfSVM(c, x);
%			trains a rbf svm with sigma^2=10 different Cs for each class
%			using SVMlight
%
% References:
%   Vapnik:The Nature of Statistical Learning Theory. Springer, 1995.
%   Burges: A Tutorial on Support Vector Machines for Pattern Recognition, 1998
%
% See also: apply_gfSVM

% Soeren Sonnenburg (2005)
% modified S. Mika (2005)
% based on work from die Guido Dornhege (2004)
% $Id: train_gfSVM.m,v 1.13 2005/06/10 10:46:06 neuro_toolbox Exp $

  
error(nargchk(3, 100, nargin));

if size(labels,1)==1,
  % Labels as +1/-1 vector. Do not do any error checking, just make sure
  % it is really only +1/-1
  labels = sign(labels);
elseif size(labels,1)==2,
  labels = [-1 1]*labels;
else
  error('gfSVM only works for binary classification tasks');
end

%% bb: setting up properties in the nice way
properties= propertylist2struct(varargin{:});
properties= set_defaults(properties, ...
                         'C',1, ...
                         'kernel','gaussian', ...
                         'implementation', 'light', ...
                         'width', 10, ...
                         'degree', 1, ...
                         'cachesize', 100, ...
                         'verbosity', 1, ...
                         'poly_inhomogene', 0, ...
                         'epsilon', 1e-5, ...
                         'C_weight', 1);

switch lower(properties.kernel),
  case 'linear',
    kernel = sprintf('LINEAR REAL %d 1.0', properties.cachesize);
  case 'poly',
    kernel = sprintf('POLY REAL %d %d %d', properties.cachesize, properties.degree, properties.poly_inhomogene);
  case 'gaussian',
    kernel = sprintf('GAUSSIAN REAL %d %f', properties.cachesize, properties.width);
  otherwise
    error('unknown kernel specified');
end

switch lower(properties.implementation)
  case 'light',
    implementation = 'LIGHT';
  case 'libsvm',
    implementation = 'LIBSVM';
  otherwise
    implementation = 'LIGHT';
end

gfSVM.properties = properties;
gfSVM.kernel_string=kernel;


switch properties.verbosity,
  case 0
	gf('send_command','loglevel ERROR');
  case 1
    gf('send_command','loglevel WARN');
  case 2
    gf('send_command','loglevel ALL');
  otherwise
    error('Unknown value for option ''verbosity''');
end
    
% $$$ gf('send_command','set_output /dev/null');
  
gf('set_features','TRAIN', data);
gf('set_labels','TRAIN', labels);

gf('send_command', sprintf('set_kernel %s',gfSVM.kernel_string));
gf('send_command', 'init_kernel TRAIN');
gf('send_command', sprintf('new_svm %s', implementation));

% Reweightning of class weights
if (properties.C_weight ~= 1) & (length(properties.C) == 1)
  % C_weight-times higher value of parameter C for smaller class
  n_size = sum(labels<0);
  p_size = sum(labels>0);
  properties.C = [1,1] * properties.C;
  if n_size<p_size
    properties.C(1) = properties.C(1) * properties.C_weight;
  else
    properties.C(2) = properties.C(2) * properties.C_weight;
  end
end
gf('send_command', sprintf('c%s', sprintf(' %f', properties.C)));
gf('send_command', sprintf('svm_epsilon %s', properties.epsilon));
gf('send_command', 'svm_train');

[gfSVM.b, gfSVM.alphas]=gf('get_svm');

gfSVM.suppvec = data(:,gfSVM.alphas(:,2)+1);
gfSVM.alphas(:,2) = 0:size(gfSVM.alphas,1)-1;

