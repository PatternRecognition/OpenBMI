function [ dat ] = prep_filterbank( dat, varargin )

if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) 
    opt=varargin{:};
end
band=opt.frequency;
num_bank=size(band,1);
for ii=1:num_bank
    CNT{ii}=prep_filter(dat , {'frequency', [band(ii,:)]});
end

dat=CNT;

end