function dat= proc_movingVariance(dat, ms, varargin)
%PROC_MOVINGVARIANCE - Moving Variance
%
%Synopsis:
% DAT= proc_movingVariance(DAT, MSEC, <METHOD='causal'>)
%
%Arguments:
% DAT    - data structure of continuous or epoched data
% MSEC   - length of interval in which the moving average is
%          to be calculated, unit [msec].
% METHOD - 'centered' or 'causal' (default).
%
%Returns:
% DAT    - updated data structure

% Author(s): Benjamin Blankertz

if length(varargin)==1,
  opt= struct('method', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'method', 'causal');

szx= size(dat.x);
x= dat.x(:,:);
n= round(ms*dat.fs/1000);
T= szx(1);
xmv= zeros(size(x));

switch(opt.method),
 case 'centered'
  nhf= floor(n/2);
  nhc= ceil(n/2);
  for k= 1:nhf,
    xmv(k,:)= var1(x(1:nhc+k-1,:));
  end
  for k= nhf+1:T-nhc+1,
    xmv(k,:)= var1(x(k-nhf:k+nhc-1,:));
  end
  for k= T-nhc+2:T,
    xmv(k,:)= var1(x(k-nhf:end,:));
  end
 
 case 'causal'
  for k= 1:n-1,
    xmv(k,:) = var1(x(1:k,:));
  end
  for k= n:T,
    xmv(k,:) = var1(x(k-n+1:k,:));
  end
  
end

dat.x= reshape(xmv, szx);



function v= var1(x)

if size(x,1)==1,
  v= zeros(1, size(x,2));
else
  v= var(x);
end
