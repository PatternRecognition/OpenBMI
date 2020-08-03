function [dat, opt] = proc_tdsep(dat, varargin)
% [dat_tdsep, tdsep_opt] = proc_tdsep(dat, <opt>)
% 
% Indendepent Component Analysis using TDsep (Ziehe & Mueller, 2008). 
% The algorithm jointly diagonalizes the signal covariance matrix as well
% as a number of time-delayed covariance matrices.
%
% Train ICA:
% [dat_ica_test, tdsep_opt] = proc_tdsep(dat_train, 'tau', tau)
%
% Apply ICA:
% dat_ica_train = proc_tdsep(dat_test, tdsep_opt)
%
% dat can be epoched or continuous. tdsep_opt contains a bias vector,
% filters and field patterns of the sources, and the used time lags.
% 
% OPT: struct or property/value list of optional properties:
%     .tau    - vector of time delays to use. Default is tau = 0:5
%
% see also: proc_pr_tdsep
% 
% Sven Daehne, 03.2011, sven.daehne@tu-berlin.de


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filters',[],...
                  'field_patters',[],...
                  'bias',[],...
                  'tau',0:5);
T = size(dat.x, 1);
n_channels = size(dat.x,2);
n_epos = size(dat.x,3);

%% train ICA
if isempty(opt.bias) || isempty(opt.filters)
    % do pca for whitening
    [pca_dat, pca_opt] = proc_pca(dat, 'whitening', 1);
    opt.bias = pca_opt.bias;
    
    % from here on we are in pca space
    % estimate the covariance matrices with different time lags
    n_lags = length(opt.tau); 
    C = zeros(n_channels, n_lags*n_channels);
    start = 1;
    for k=1:n_lags
        stop = k*n_channels;
        C(:,start:stop) = lagged_covariance_matrix(pca_dat.x, opt.tau(k));
        start = stop + 1;
    end
    opt.C = C;
    if n_lags == 2
        [W,D] = eig(C(:,1:n_channels), C(:,(n_channels+1):end));
    else
        % joint diagonalization -> this line is taken from proc_pr_tdsep
        [W,D] = utils_proc_pr_tdsep_jdiag(C,0.00000001);
    end
    opt.D = D;
    % project the ica filters and field patterns from pca space to sensor space
    opt.filters = pca_opt.filters * W;
    opt.field_patterns = pca_opt.field_patterns * inv(opt.filters)';
end

%% apply ICA
if not(length(opt.bias) == n_channels)
    error('Dimension of bias must equal the number of channels!')
end
% make sure opt.bias is a row vector
if size(opt.bias, 1) > size(opt.bias,2)
    opt.bias = opt.bias';
end

B = squeeze(repmat(opt.bias, [T,1,n_epos]));
dat.x = dat.x - B;
dat = proc_linearDerivation(dat, opt.filters);





function C = lagged_covariance_matrix(X, tau)

n_samples = size(X,1);
n_channels = size(X,2);
n_epos = size(X,3);
C = zeros(n_channels, n_channels);
mean_flag = 0; % the data is mean free already
for e=1:n_epos
    x = squeeze(X(:,:,e));
    C = C + utils_proc_pr_tdsep_cor2(x,tau,mean_flag);
end
C = C/(n_samples*n_epos);