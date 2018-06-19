% Mutual Information based Best Individual Feature (MIBIF)
% Feature selection algorithm - ref. Filter bank common spatial filter

function [ dat_out ] = func_MIBIF( dat_in, varargin )

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:};
end

num_feat=opt.nFeatures;
dat_out=dat_in;
dat=dat_in.x;
label=dat_in.y_dec; 

mutual_info=func_mutual_information(dat,label);
dat_out.mutual_info=mutual_info;

end

