function C = train_mkl(dat,labels,dimensions, c,varargin);
%TRAIN_MKL trains the multiple kernel learning classifier by Gunnar Raetsch and Soeren Sonnenburg's genefinder
%
% usage:
% C = train_mkl(dat,labels,dimensions,c,kernel1,...);
%
% input:
%   dat      data (in columns)
%   labels   labels a logical array (only two class)
%   dimensions  cell array of dimensions for each feature
%   c        regularisation constant
%   kernel*  is a cell array with entries
%            feature (can be an array or empty which means all feature)
%            kernelname (name of the kernel (known by gf!!!)
%            kernelparams (further params of the kernel)
%
%
% output:
%   C        classifier with fields:
%
%
% See gf for documentation and more informations
% Guido Dornhege, 19/04/2005

clear gf

gf('send_command', 'clean_kernels') ;

if size(labels,1)>2
  error('only two class');
end

if size(labels,1)==2
  labels = [-1 1]*labels;
end

if size(dat,2)~=length(labels)
  error('length of data and labels does not fit');
end

if nargin<=2
  dimensions = {size(dat,1)};
end

if ~iscell(dimensions)
  if nargin>=3
    varargin = cat(2,{c},varargin);
  end
  c = dimensions;
  dimensions = {size(dat,1)};
end

if ~exist('c','var') | isempty(c)
  c = 1;
end

if length(varargin)==0
  varargin = {{[],'POLY',1,0,1}};
end


cache_size = ceil(16*size(labels,2)^2/1024/1024);

nF = length(dimensions);
gf( 'clean_features', 'TRAIN' );
gf( 'clean_features', 'TEST' );
feature = cell(1,nF);
start = 0;
for i = 1:nF
  feature{i} = dat(start+1:start+prod(dimensions{i}),:);
  start = start+prod(dimensions{i});
end




% kernels
fea = [];
ker = {};

for i = 1:length(varargin)
  kernel = varargin{i};
  fe = kernel{1};
  nam = kernel{2};
  kepa = kernel(3:end);
  kep = '';
  for i = 1:length(kepa)
    if kepa{i}==round(kepa{i})
      kep = sprintf('%s %d',kep,kepa{i});
    else
      kep = sprintf('%s %f',kep,kepa{i});
    end
  end
  kepa = kep;

  if isempty(fe)
    fe = 1:nF;
  end
  fe = intersect(fe,1:nF);
  fea = cat(2,fea,fe);
  for j = 1:length(fe)
    ker = cat(2,ker,{sprintf('add_kernel 1 %s REAL %d %s',nam,cache_size,kepa)});
  end
end

for i = 1:length(fea)
  gf('add_features','TRAIN', feature{fea(i)});
end

  
gf('set_labels','TRAIN', labels);
gf('send_command', 'new_svm LIGHT');
gf('send_command', 'use_linadd 0');
gf('send_command', 'use_mkl 1');
gf('send_command', 'use_precompute 1');
gf('send_command', 'mkl_parameters 1e-3 0');
gf('send_command', 'svm_epsilon 1e-3');

gf('send_command', sprintf('set_kernel COMBINED %d', cache_size));

for i = 1:length(ker)
  gf('send_command', ker{i});
end



gf('send_command', sprintf('c %1.2e', c)) ;
gf('send_command', 'init_kernel TRAIN');
gf('send_command', 'svm_train');



C = struct('dimensions',{dimensions});

[C.b,alp]=gf('get_svm') ;
C.kernel = ker;
C.feature = fea;

C.SV = dat(:,alp(:,2)+1);
C.w = alp(:,1);
C.kw=gf('get_subkernel_weights');
C.alpha = (0:size(alp,1)-1)';
C.labels = labels(alp(:,2)+1);


gf('send_command', 'clean_kernels') ;
gf( 'clean_features', 'TRAIN' );
gf( 'clean_features', 'TEST' );
