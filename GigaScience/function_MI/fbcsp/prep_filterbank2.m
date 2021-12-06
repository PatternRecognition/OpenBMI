function [ dat ] = prep_filterbank2( dat, varargin ) 

if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:};
end
band=opt.frequency;
num_bank=size(band,1);
for ii=1:num_bank
    CNT{ii}=dat;
    [b,a]= butter(2, [band(ii,:)]/(1000/2));
    CNT{ii}.x=filter(b,a,dat.x);
end
dat=CNT;
end