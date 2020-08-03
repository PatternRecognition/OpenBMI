function out = sigmoid_function(out,varargin);
% out = sigmoid_function(out,A,B)
% 
% This function can be used to transform a classifier output
% into probabilistic values, by using the function
% f(x) = 1/(1+exp(x*A+B))
% 
% A and B can be learned with sigmoid_training(out,label).
% 
% Alternatively:
% out = sigmoid_function(out,[],A,B)
% for calling it as postprocessing in bbci_bet_apply.
% A,B can be double arrays of equal length as out.
% out can be either double or cell array.

% kraulem01/08

A = varargin{nargin-2};
B = varargin{nargin-1};
for ii = 1:length(out)
  if iscell(out)
    out{ii} = 1./(1+exp(out{ii}*A(ii)+B(ii)));
  else
    out(ii) = 1./(1+exp(out(ii)*A(ii)+B(ii)));
  end
end
return


