function [Cstar, gamma, T]= clsutil_fastShrinkage(X, varargin)
% IN CONSTRUCTION

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'target', 'B', ...
                  'gamma', 'auto', ...
                  'verbose', 0);

if isequal(opt.gamma, 'auto'),
  gamma= NaN;
elseif isreal(opt.gamma),
  gamma= opt.gamma;
else
  error('value for OPT.gamma not understood');
end
  
%% Empirical covariance
[p, n]= size(X);
Xn= X - repmat(mean(X,2), [1 n]);
S= Xn*Xn';
Xn2= Xn.^2;

V= 1/(n-1) * (Xn2 * Xn2' - S.^2/n);
T= mean(diag(S))*eye(p,p);
gamma= n * sum(sum(V)) / sum(sum((S - T).^2));

%% Handle special cases
if gamma>1,
  if opt.verbose,
    warning('gamma forced to 1');
  end
  gamma= 1;
elseif gamma<0,
  if opt.verbose,
    warning('gamma forced to 0');
  end
  gamma= 0;
end

Cstar= (gamma*T + (1-gamma)*S ) / (n-1);
