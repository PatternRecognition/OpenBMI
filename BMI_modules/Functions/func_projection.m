function [ out ] = func_projection( dat, w )
% func_projection:
%
%   This function projects the data with w(a vector or a matrix).
% 
% Example:
%   dat=func_projection(dat, csp_w);
% 
% Input:
%     dat - Data structure or data itself
%     w   - Projection vector
% Returns:
%     fv  - Feature vector
% 

if iscell(w)
    w=w{:};
end

if isstruct(dat)
    if isfield(dat, 'x')
        tDat=dat.x;
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
    end
    % stack
    if isfield(dat, 'stack')        
        c = mfilename('fullpath');
        c = strsplit(c,'\');
        dat.stack{end+1}=c{end};
    end
    
else
    out=in;
end


end

