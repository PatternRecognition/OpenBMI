function [ cf_out ] = func_predict( in, varargin )
%PROC_APPLY_CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

if iscell(varargin)
    varargin=varargin{:};
end
opt=varargin{1};

if isstruct(in)
dat=in.x;
else
    dat=in;
end

switch lower(opt.classifier)
    case 'lda'     
        cf_out= real( dat'*opt.cf_param.w+opt.cf_param.b);% + repmat(opt.cf_param.b, [1 size(fv.x,1)]) );
end

end

