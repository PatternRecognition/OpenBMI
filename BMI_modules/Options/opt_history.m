function dat =  opt_history(dat,funcname,opt)

if nargin == 2
    opt = struct();
end

if ~isfield(dat,'history')
    dat.history = {funcname,opt};
else
    dat.history(end+1,:) = {funcname,opt};
end
end