function [ out ] = func_projection( dat, w, varargin )
%PROC_PROJECTION Summary of this function goes here
%   Detailed explanation goes here
if isstruct(dat)
    if isfield(dat, 'x')
        tDat=dat.x;
    elseif isfield(dat, 'cnt')
        tDat=dat.cnt;
    else
        error('parameter error of dat.x')
    end
else % no structure
    tDat=dat;
end

if ndims(tDat)==2,
    in= tDat*w;
else
    sz= size(tDat);
    in= reshape(tDat, sz(1)*sz(2), sz(3));
    in= in*w;
    in= reshape(in, [sz(1) sz(2) size(in,2)]);
end

if isstruct(dat)
    out=dat;
    if isfield(dat, 'x')
        out.x=in;
    elseif isfield(dat, 'cnt')
        out.cnt=in;
    end
    % stack
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    dat.stack{end+1}=c{end};
    
else
    out=in;
end


end

