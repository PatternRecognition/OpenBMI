function [A,B] = sigmoid_training(out,label,varargin)
% [A,B] = sigmoid_training(out,label)
% 
% A and B can be used to transform the classifier output
% into probabilistic values, by using the function
% f(x) = 1/(1+exp(x*A+B))
% 
% IN:    out - classifier output
%        label-training data of the labels.
% OUT:   A,B - constituting parameters for the sigmoid function.
%
% Code taken from: J. Platt, 'Probabilistic Outputs for SVM and
% Comparisons to Regularized Likelihood Methods', MIT Press (99).

% kraulem01/08

if size(label,1)>1
  label = [-1 1]*label;
end
prior1 = sum(label>0);
prior0 = sum(label<0);
A = 0;
B = log((prior0+1)/(prior1+1));
hiTarget = (prior1+1)/(prior1+2);
loTarget = 1/(prior0+2);
lambda = 1e-3;
olderr = 1e300;
% array to store the current estimate of probability of examples:
pp = ones(1,length(out))*(prior1+1)/(prior0+prior1+2);
count = 0;
for it = 1:100
  % First compute Hessian & gradient of error function
  % w.r.t. A & B
  t(find(label>0)) = hiTarget;
  t(find(label<0)) = loTarget;
  d1 = pp-t;
  d2 = pp.*(ones(1,length(label))-pp);
  a = sum(out.*out.*d2);
  b = sum(d2);
  c = sum(out.*d2);
  d = sum(out.*d1);
  e = sum(d1);
  % If gradient is really tiny, then stop
  if abs(d)<1e-9 & abs(e)<1e-9
    break
  end
  oldA = A;
  oldB = B;
  err = 0;
  % Loop until goodness of fit increases
  while true
    det = (a+lambda)*(b+lambda)-c*c;
    if det==0
      % if determinant of Hessian is zero, increase stabilizer
      lambda = lambda*10;
      continue
    end
    A = oldA+((b+lambda)*d-c*e)/det;
    B = oldB+((a+lambda)*e-c*d)/det;
    % Now compute the goodness of fit.
    err = 0;
    for ii = 1:length(out)
      p = 1/(1+exp(out(ii)*A+B));
      pp(ii) = p;
      % At this step, note that log(0) = -inf.
      err = err-t*log(p)-(1-t)*log(1-p);
    end
    if err<olderr*(1+1e-7)
      lambda = lambda*.1;
      break
    end
    % error did not decrease: increase stabilizer by factor 10
    % and try again.
    lambda = lambda*10;
    if lambda>=1e6
      % something is wrong. Give up.
      break
    end
  end
  diff = err-olderr;
  scale = .5*(err+olderr+1);
  if diff>(-1e-3)*scale & diff<(1e-7)*scale
    count = count+1;
  else
    count = 0;
  end
  olderr = err;
  if count==3
    break
  end
end


