function [nat1Site,nat2Site,nat1Cavity,nat2Cavity,invL,evidence] = gaussProc_classificationEP(K,y,opt);
% gaussProc_classificationEP - Expectation Propagation for GP classification
%
% Synopsis:
%   [nat1Site,nat2Site,nat1Cavity,nat2Cavity] = gaussProc_classificationEP(K,y,opt)
%   [nat1Site,nat2Site,nat1Cavity,nat2Cavity,invL,evidence] = gaussProc_classificationEP(K,y,opt)
%   
% Arguments:
%  K: [N N] kernel matrix
%  y: [1 N] vector, +1 resp -1 class labels
%  opt: GP options structure. Used fields are only opt.EPiterations,
%      opt.EPtolerance and opt.verbosity
%   
% Returns:
%  nat1Site: [N 1] vector. Natural parameter 1 of site distributions
%  nat2Site: [N 1] vector. Natural parameter 2 of site distributions
%  nat1Cavity: [N 1] vector. Natural parameter 1 of cavity distributions
%  nat2Cavity: [N 1] vector. Natural parameter 2 of cavity distributions
%  invL: [N N] matrix. Cholesky factor of inverse approximate covariance
%      matrix
%  evidence: Scalar. Log marginal likelihood with EP approximation
%   
% Description:
%   Subroutine of evidence_classification
%   
%   
% Examples:
%   
%   
% See also: gaussProc_classification,gaussProc_classificationEPparams
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2006
% $Id: gaussProc_classificationEP.m,v 1.3 2006/08/02 17:25:28 neuro_toolbox Exp $


m = size(K,1);

% Need to do an awful hack here: often when kernel parameters go to bizarre
% values, EP does not work. Each call of EP again starts it zero, the optimum can not
% be found anymore. This leads to zeros for the site parameters. 
% Now: keep the site parameters from the last call to classificationEP
% and start from there.
persistent site1init site2init cavity1init cavity2init
if isempty(site1init) | isempty(site2init) | isempty(cavity1init) | ...
      isempty(cavity2init) | length(site1init)~=m,
  % Site parameters (natural parameterisation)
  site1init = zeros(m,1);     % nat1 = sigma^{-2}
  site2init = zeros(m,1);     % nat2 = mu * sigma^{-2}
  cavity1init = zeros(m,1);    % The means and variances of the approximate
  cavity2init = zeros(m,1);    % cavity distributions
  % The Gaussian approximation
  A = K;                     % The covariance matrix and
  mu = zeros(m,1);           % the mean of the Gaussian approximations
else
  % OK, so we do have old values for the site parameters that are of
  % appropriate length. Look at the marginal likelihood & try to guess
  % whether it makes sense to use them
  [A,mu,invL,evidence] = ...
      gaussProc_classificationEPparams(K,site1init,site2init,cavity1init,cavity2init,y);
  % In very rare cases, when working "on the edge of computability",
  % the old values may be bizarre/zero/inf already. Select new parameters
  % if this should have happened (thanks to Ryota Tomioka for that hint)
  if evidence>(m*log(2)) | isinf(evidence) | isnan(evidence),
    % Negative marginal likelihood is larger than what we would have
    % obtained with using zeros: Use the zeros as initial values
    site1init = zeros(m,1);     % nat1 = sigma^{-2}
    site2init = zeros(m,1);     % nat2 = mu * sigma^{-2}
    cavity1init = zeros(m,1);    % The means and variances of the approximate
    cavity2init = zeros(m,1);    % cavity distributions
    A = K;                     % The covariance matrix and
    mu = zeros(m,1);           % the mean of the Gaussian approximations
  end
end
nat1Site = site1init;       % nat1 = sigma^{-2}
nat2Site = site2init;       % nat2 = mu * sigma^{-2}
nat1Cavity = cavity1init;    % The means and variances of the approximate
nat2Cavity = cavity2init;    % cavity distributions


%
% EP Main Loop
%

iteration = 0;
converged = false;

while (iteration < opt.EPiterations) & (converged == false)
  iteration = iteration + 1;
  
  oldnat1Site = nat1Site;
  oldnat2Site = nat2Site;
  
  % Iterate over all site parameters
  for i = 1:m;  % randperm(m)

    % Compute parameters of cavity distribution
    nat1Cavity(i) = (A(i,i)^(-1) - nat1Site(i));
    nat2Cavity(i) = (mu(i)/A(i,i) - nat2Site(i));
    
    if nat1Cavity(i) > 0
      
      % Compute moments
      var = 1/nat1Cavity(i); 
      mu = nat2Cavity(i)/nat1Cavity(i);
      z = y(i)*mu/sqrt(1+var);
      m0 = 0.5 * erfc(-z/sqrt(2));
      m1 = mu + var *y(i)* exp((z.^2)/(-2)) / (sqrt(2*pi)) / (m0*sqrt(1+var));
      m2 = 2 * mu * m1 - mu^2 + var - (var^2 *z* exp((z.^2)/(-2))/(sqrt(2*pi)))/(m0*(1+var));

      % Update site parameters
      m2r = (m2-m1^2);
      nat1Site(i) = 1/m2r - nat1Cavity(i);
      nat2Site(i) = m1/m2r - nat2Cavity(i);
      
      % Update A and mu
      delta =  nat1Site(i) - oldnat1Site(i);
      A = A - A(:,i) * (delta./(A(i,i) * delta + 1)) * A(:,i)';
      mu = A * nat2Site;
    else
      % Emergency exit: Set cavity parameter to 0, to avoid problems when
      % computing sqrt(1+1./nat1Cavity) in the outer loop
      nat1Cavity(i) = 0;
    end
  end
  if opt.verbosity>2,
    fprintf('EP sweep %i: maximum change = %f\n', iteration, ...
            max(abs(oldnat2Site - nat2Site)));
  end
  if  max(abs(oldnat2Site - nat2Site)) < opt.EPtolerance,
    converged = true;
  end
  % Compute A from scratch, since rank 1 updates can destroy precision
  [A,mu,invL,evidence] = ...
      gaussProc_classificationEPparams(K,nat1Site,nat2Site,nat1Cavity,nat2Cavity,y);
end

if ~converged,
  if opt.verbosity>0,
    fprintf('EP failed to converge: change = %f \n', max(abs(oldnat2Site - nat2Site)));
  end
end

site1init = nat1Site;
site2init = nat2Site;
cavity1init = nat1Cavity;
cavity2init = nat2Cavity;
