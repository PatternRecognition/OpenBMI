function dat=func_ar(dat, order, varargin)
% func_ar (Feature extraction) :
% 
% This function calculates the autoregression(AR) parameter.
% 
% Example:
% [out] = func_ar(dat, 7, {'method','arburg'})
% 
% Returs:
%     dat    - Data structure, segmented
%     order  - Order of AR setting
% Option: models for obtatining AR parameter
%     method - 'aryule'(default), 'arburg', 'arcov', 'armcov' 
% 

opt=opt_cellToStruct(varargin{:});
opt=struct('method',opt.method);

if isempty(dat)
    warning('[OpenBMI] Warning! data is empty.');
end

if isempty(order)
    warning('[OpenBMI] Order is not exist.');
end

if isempty(opt.method) %method selection
   opt.method='aryule';
end

[T, nEvents , nChans]= size(dat.x);

temp_ar= [];
for i= 1:nChans*nEvents,
  ar= feval(opt.method, dat.x(:,i), order);
  temp_ar(:,i)= ar(2:end)';
end

dat.x= temp_ar;
