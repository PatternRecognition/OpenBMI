function [SMT, CSP_W, CSP_D]=func_csp_filterbank(SMT_off,varargin)

if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:})
    opt=varargin{:}
end

dat=SMT_off;
CSPFilter=opt.nPatterns;

for ii=1:length(dat)
    [SMT{ii}, CSP_W{ii}, CSP_D{ii}]=func_csp(dat{ii},{'nPatterns', CSPFilter});
end


end