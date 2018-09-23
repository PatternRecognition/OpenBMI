function [out] = func_projection(dat, w)
% func_projection:
%     Computes common spatial patterns (CSP) mehtods.
% 	  this version supports binary classes. Multi class csp version will be
% 	  updated soon.
%
% Example:
%     [out] = func_projection(SMT, {'n_patterns', [3]});
%
% Input:
%     dat - Data structure of epoched
%
% Returns:
%     out -  

if nargin < 2
    error('OpenBMI: Input data should be specified');
end

if iscell(w)
    w = w{:};
end

if isstruct(dat)
    if isfield(dat, 'x')
        tDat = dat.x;
    else
        error('OpenBMI: Data must have fields named ''x''');
    end
else % no structure
    tDat = dat;
end

if ismatrix(tDat)
    in = tDat * w;
elseif ndims(tDat) == 3
    sz = size(tDat);
    in = reshape(tDat, sz(1) * sz(2), sz(3));
    in = in * w;
    in = reshape(in, [sz(1) sz(2) size(in, 2)]);
else
    error('OpenBMI: Check the dimensions of data');
end

if isstruct(dat)
    out = dat;
    if isfield(dat, 'x')
        out.x = in;
    end
    out = opt_history(out, 'func_projection', struct([]));
else
    out = in;
end

end