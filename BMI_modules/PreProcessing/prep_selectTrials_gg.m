function [ out ] = prep_selectTrials( dat, varargin )
%PROC_SELECT_TRIALS Summary of this function goes here
%   Detailed explanation goes here
if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:};
end
%
if ~isfield(opt,'index')
    warning('OpenBMI: the essential parameter "index" is missing');
    return;
end

in=dat;
if isfield(in, 't')
    in.t=dat.t(:,opt.index,:);
end
if isfield(in, 'y_dec')
    in.y_dec=dat.y_dec(opt.index);
end
if isfield(in, 'y_logic')
    in.y_logic=dat.y_logic(:,opt.index);
end
if isfield(in, 'y_class')
    in.y_class=dat.y_class(opt.index);
end

if isfield(in, 'y_class')
    in.y_class=dat.y_class(opt.index);
end
if isfield(in, 'x')
    if  ndims(dat.x)==3
        in.x=dat.x(:,opt.index,:);
    elseif ndims(dat.x)
        % do nothing
    end
    
end
out=in;
end

