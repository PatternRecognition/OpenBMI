% F cumulative distribution function
% (not fully evaluated: n=inf not treated)
%
function [y] = proc_fcdf(x,m,n)

si = size(x);

  if(m>0&n>0)
    if(prod(double(isfinite(x))))
      if(prod(double(x>0)))
        y = betainc(x./(x + n./m), m/2, n/2);
      else
        y = zeros(si);
      end
    else
      y = ones(si);
    end
  else
    y = NaN*ones(si);
  end

end
