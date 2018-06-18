function [ out ] = func_projection_filterbank( dat, w )

CSP_W=w;

for ii=1:length(CSP_W)
    SMT{ii}=func_projection(dat{ii}, CSP_W{ii});
end

out=SMT;
end