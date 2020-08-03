function [out]= proc_mulevel(dat, mu)
%[out]= proc_mulevel(dat, mu)

out= dat;

for icl = 1:size(dat.x, 2)
%    out.x(:, icl, :) = max(0, min(1, (out.x(:, icl, :)-mu.min(icl))/(mu.max(icl)-mu.min(icl))));
  out.x(:, icl, :) = (out.x(:, icl, :)-mu.min(icl))/(mu.max(icl)-mu.min(icl));
end

