function dat=func_ar(dat, order, varargin)
%Calculating the auto regression parameter / aar 
%In dat      -     input the data structure of OpenBMI segementation original data
%   order    -     order of AR setting
%   varadgin -     model selection for obtatining AR parameter 
%                  deafualt is 'aryule';
%                  model : 'arburg', 'arcov', 'armcov' 

%out dat     -     data structure of otanined ar parameter in OpenBMI sturcture

% Example code func_AR(dat, 7, {'method','arburg'})
% Example code

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
