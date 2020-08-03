function C = train_mkl_xval(dat,labels,dimensions, c,varargin);
%TRAIN_MKL_XVAL is a wrapper around train_mkl to be used with xvalidation
%
% usage:
% C = train_mkl_xval(dat,labels,dimensions,c,kernel1,kpars1,kernel2,kpars2...);
%
% input:
%   dat         data (in columns)
%   labels      labels a logical array (only two class)
%   dimensions  cell array of dimensions for each feature
%   c           regularisation constant
%   kernel*     is a cell array with entries
%               feature (can be an array or empty which means all feature)
%               kernelname (name of the kernel (known by gf!!!)
%               kernelparams (further params of the kernel)
%   kpars*      parameter, value
%
%
% output:
%    C           classifier
%
% This function is necessary because  xvalidation cannot change  
% parameters inside cell arrays (some of the parameters to train_mkl 
% need to be given in cell arrays).
%
% Syntax is similar to that of train_mkl, with two important differences: 
%
% 1. Parameters must not be omitted (if you want to use the defaults of
% train_mkl, enter them explicitly).
%
% 2. Each kernel may be followed by one or more parameter, value pairs. 
% If so, parameters in the cell array kernel* are overwritten with the 
% value(s) given in the parameter, value pair(s). Paramters for the kernel 
% inside the cell array must not be omittet, you need to specify them, 
% so they can be overwritten later.
%
%
%
% Example
%
% fv = proc_catFeatures(fvFOO, fvBAR);
% dimensions = {  size(fvFOO.features,1),size(fvBAR.features,1) };
% kernels =   {  {1:size(fvFOO.features,1), 'GAUSSIAN', 10, }, 3, '*lin',...                   
%   {(size(fvFOO.features,1)+1):dimensions, 'GAUSSIAN', 10  }, 3, '*lin'};
%
% model.classy = {'mkl_xval', dimensions, '*lin', kernels{:}};
%
% model.param(1) = struct('index', 3, 'value', [0.01, 0.1, 1, 10]);
% model.param(2) = struct('index', 6, 'value', [0.01, 0.1, 1, 10]); 
% model.param(3) = struct('index', 9, 'value', [0.01, 0.1, 1, 10]);
%
% % set CVopt etc.       
%
% [aoc, aoc_std, out, memo] = xvalidation(fv, model, CVopt);
%
%
% Timon Schroeter, 25/05/2005


if nargin<5
  error('not enough arguments, you want to read the documentation');
end

if size(labels,1)>2
  error('only two class');
end

if size(labels,1)==2
  labels = [-1 1]*labels;
end

if size(dat,2)~=length(labels)
  error('length of data and labels does not fit');
end

if ~iscell(dimensions)
  error('dimensions needs to be a cell');  
end


if length(varargin)==0
  error('you need to specify kernels & their parameters');
end

i = 1;
argout = {};

while i <= length(varargin)  
  if ~iscell(varargin{i})
    error('kernels need to be given as cell arrays');
  end  

  kernelout = varargin{i};
  
   if i+2 <= length(varargin)
    if ~iscell(varargin{i+1})
      if ~iscell(varargin{i+2})
        parameter = varargin{i+1};
        value = varargin{i+2};
        if parameter > length(kernelout);
          error('parameter is larger than the numer of parameters you specified for this kernel');    
        end
        %error('stop, whats going on?');
        kernelout{parameter} = value;
        i = i+2;
      else
        error('parameters and values must be given in pairs');  
      end    
    end
  end   
  argout = cat(2, argout, {kernelout});
  i = i+1;
end

C = train_mkl(dat,labels,dimensions, c,argout{:});
