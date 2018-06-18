function [ out ] = func_featureExtraction_filterbank( dat, varargin )

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) 
    opt=varargin{:};
end

for ii=1:length(dat)
    FT{ii}=func_featureExtraction(dat{ii}, {'feature',opt.feature});
end

out=FT;
end