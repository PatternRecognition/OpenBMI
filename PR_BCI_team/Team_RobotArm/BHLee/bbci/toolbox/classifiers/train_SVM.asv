function SVM = train_SVM(xTr,yTr,C,arg4,varargin);
% train_SVM - trains a Suport Vector Machine (SVM) using the genefinder program
%
% Synopsis:
%	SVM = train_SVM(xTr,yTr,<C=1,kernel='linear',kernelparams={}>);
%	SVM = train_SVM(xTr,yTr,<C=1,cost=1,kernel='linear',kernelparams={}>);
%
% Arguments:
%	xTr: [m,n] m feature vectors of length n
%	yTr: [1,m] m labels (+/-1 or logical)
% 	C: Regularization parameter C>0
%
% Returns:
%	SVM: a trained SVM with fields:
%             .SV: [l,n] l support vectors
%             .SVlab:  [1,l] labels of the support vectors
%             .alphas: [1,l] lagrange multiplier for each support vector
%             .b: bias
%             .kernel: kernel with parameters as string
%
% Properties:
%	'implementation': Which SVM implementation to use:
%			light: SVMLight (see http://svmlight.joachims.org) (default)
%			libsvm: libSVM (see http://www.csie.ntu.edu.tw/~cjlin/libsvm )
% 	'cost': A factor which describes how missclassification 
%		  in the second class is more expensive than in the first 
%		  class, i.e. if the first class is three times as big as 
%		  the second one you should use 3 here.
%		  using '*' selects the optimal cost for the class priors
%		  as found in the training set.
% 	'kernel': the name of the kernel to be used
%			linear: K(u,v) = u'*v;
%			poly: K(u,v) = (u'*v)^d when homogeneous
%				  K(u,v) = (u'*v+1)^d when inhomogeneous
%			gaussian: K(u,v) =  exp(-||u-v||^2/width)
%	'kernelcachesize': size of the kernel cache in megabytes (default 100MB)
% 	'kernelparam': further params for the kernel, e.g. for gaussian
%             the kernel width (=1) or for poly the degree and
%             shift ({2, 0})
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
%   x = [randn(10,50-0.5; randn(10,50)];
%	y = [-ones(1,50) ones(1,5)];
%	s1 = train_SVM(x, y, 10, { kernel='linear', kernelparms='5' });
%	s2 = train_SVM(x, y, 10, { kernel='poly', kernelparms='5' });
%	s3 = train_SVM(x, y, 10, { kernel='gaussian', kernelparms='2' });
%   
%	trains three SVMs s1,s2,s3 using a
%		1) linear
%		2) polynomial kernel of degree 5
%		3) rbf kernel with width 2
%	regularizing with C=10.
%	
% References:
%   Vapnik:The Nature of Statistical Learning Theory. Springer, 1995.
%   Burges: A Tutorial on Support Vector Machines for Pattern Recognition, 1998
%
% See also: apply_SVM
% Guido Dornhege (2003)
% Soeren Sonnenburg (2004)
% Benjamin Blankertz (2005): call train_gfSVM
% $Id$

persistent t0

give_warning= 1;
if ~isempty(t0) & etime(clock, t0)<5*60,
  give_warning= 0;
end
if give_warning,    
  warning('function is obsolete, use train_gfSVM')
  t0= clock;
end


if ~exist('kernel','var') | isempty(kernel)
  kernel = 'linear';
end

if ~exist('C','var') | isempty(C)
  C = 1;
end

if size(yTr,1)>2
  error('SVM only works for binary classification tasks');
end

if size(yTr,1)==2
  yTr = [-1 1]*yTr;
end

if ischar(arg4) & arg4=='*',
  arg4= sum(yTr==-1)/sum(yTr==1);
end

if isnumeric(arg4),
  fp= min(find(yTr>0));
  xTr(:,[1 fp])= xTr(:,[fp 1]);
  yTr(:,[1 fp])= yTr(:,[fp 1]);
  arg4= 1/arg4;
  C = [C, C*arg4];
  kernel = varargin{1};
  varargin = {varargin{2:end}};
else
  kernel = arg4;
end


%% begin change by bb: call new function train_gfSVM
switch lower(kernel),
 case 'linear',
 case 'gaussian',
  if length(varargin)>=1
    param= {'width', varargin{1}};
  else
    param= {'width', 1};
  end
 case 'poly',
  if length(varargin)>=2
    param= {'degree',varargin{1}, 'poly_inhomogene',varargin{2}};
  elseif length(varargin)==1
    param= {'degree',varargin{1}, 'poly_inhomogene',0};
  else
    param= {'degree',2, 'poly_inhomogene',0};
  end
 otherwise
  error('kernel unknown');
end

SVM= train_gfSVM(xTr, yTr, 'C',C, 'kernel',kernel, 'cachesize',100, ...
                 param{:});

return

%% end change by bb




%% old, dead code:


switch lower(kernel),
 case 'linear',
  kernel = 'LINEAR REAL 100';
 case 'gaussian',
  if length(varargin)>=1
    kernel = sprintf('GAUSSIAN REAL 100 %f',varargin{1});
  else
    kernel = 'GAUSSIAN REAL 100 1';
  end
 case 'poly',
  if length(varargin)>=2
    kernel = sprintf('POLY REAL 100 %d %d',varargin{1:2})
  elseif length(varargin)==1
    kernel = sprintf('POLY REAL 100 %d 0',varargin{1});
  else
    kernel = 'POLY REAL 100 2 0';
  end
 otherwise
  kernel = [upper(kernel) ' REAL 100 '];
  for i = 1:length(varargin)
    if isnumeric(varargin{i})
      kernel = [kernel num2str(varargin{i}) ' '];
    else
      kernel = [kernel varargin{i} ' '];
    end
  end
end

SVM.kernel = kernel;

gf('set_features','TRAIN',xTr);
gf('set_labels','TRAIN',yTr);

gf('send_command', sprintf('set_kernel %s',SVM.kernel));
gf('send_command', 'init_kernel TRAIN');
gf('send_command', sprintf('new_svm %s',SVM.implementation));
gf('send_command', ['c ' num2str(C)]);
gf('send_command', 'svm_train');


[SVM.b, SVM.alphas]=gf('get_svm');

SVM.SV = xTr(:,SVM.alphas(:,2)+1);
SVM.SVlab = yTr(SVM.alphas(:,2)+1);
SVM.alphas(:,2) = 0:size(SVM.alphas,1)-1;
