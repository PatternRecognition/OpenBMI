function [fv, P, b] = proc_pr_tdsep(fv, retain, varargin);
%Ziehe's TDSEP algorithm (ICA)
%fv.x --> fv.x*P - b
%
%[fv, P, b]= proc_pr_tdsep(fv, retain, <opt>)
%
% IN  fv     - struct of feature vectors. fv.x is a matrix [T n] for n
%              sensors, T timesteps
%     retain - threshold for determining how many features to retain,
%              depends on opt.policy (default: keep all)
%     opt    propertylist or struct of options:
%      .policy - one of 'number_of_features' (default), 'perc_of_features',
%                'perc_of_score': determines the strategy how to choose
%                the number of features to be selected
%      .tau    - vector of time lag values
%
% OUT  fv    - struct of new (poss. reduced) feature vectors. fv.x 
%              is a [T retain] matrix
%      P     - projection matrix
%      b     - mean value before centering
% 

% fcm 16jul2004 
% fcm 08apr2005: changed all fv.x to fv.x' to meet the toolbox standard
% Anton Schwaighofer, Sep 2005
% $Id: proc_pr_tdsep.m,v 1.6 2005/09/19 15:12:28 neuro_toolbox Exp $



% the default settings
defopt.policy = 'number_of_features';
defopt.tau = [0 1];
defopt.whitening = 1;

% read the parameters
if ~exist('retain','var')|isempty(retain),
  opt.policy = 'perc_of_features';
  retain = 100;
end;
if length(varargin)==1,
  if isstruct(varargin{1}),
    opt = varargin{1};
  else
    error('Optional parameters should always go by pairs or as fields of a struct');
  end;
elseif length(varargin)>1,
  if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs or as fields of a struct');
  else
    opt = propertylist2struct(varargin{:});
  end;
end;
if ~exist('opt','var'),
  opt = defopt;
else
  opt = set_defaults(opt, defopt);
end;

% perform PCA for dimensionality reduction, whitening. PCA expects the
% data the other way around: [dim Nsamples]
[X, PCAdata] = proc_pr_pca(fv.x', retain, opt);
X = X';
p = size(fv.x,1);

N = length(opt.tau);
if N == 2,
    % solve directly as a general eigenvalue problem
    M1 = utils_proc_pr_tdsep_cor2(X,opt.tau(2));
    [A,D] = eig(M1);
else
    t=1;          % compute correlation matrices
    %   for sel = 1:N,
    %     M(:,t*p+1:((t+1)*p)) = utils_proc_pr_tdsep_cor2(X,opt.tau(sel));
    %     t=t+1;
    %   end
    n_chans = size(X,2);
    start = 1;
    for k = 1:N
        stop = start+n_chans-1;
        M(:,start:stop) = utils_proc_pr_tdsep_cor2(X,opt.tau(k));
        start = stop+1;
    end
    % joint diagonalization
    [A,D] = utils_proc_pr_tdsep_jdiag(M,0.00000001);
end

P = (inv(A)*PCAdata.P)';
fv.x = X*inv(A)';
b = (inv(A)*PCAdata.b)';
