function [dat, opt] = proc_jade(dat, varargin)
% [dat_jade, jade_opt] = proc_jade(dat, <opt>)
% 
% Indendepent Component Analysis using JADE (a cumulant-based approach).
%
% Train ICA:
% [dat_ica_test, jade_opt] = proc_jade(dat_train, 'tau', tau)
%
% Apply ICA:
% dat_ica_test = proc_jade(dat_test, jade_opt)
%
% dat can be epoched or continuous. jade_opt contains a bias vector,
% filters and field patterns of the sources, and the used time lags.
% 
% OPT: struct or property/value list of optional properties:
%     
%
% see also: proc_pr_jade
% 
% Sven Daehne, 08.2011, sven.daehne@tu-berlin.de


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filters',[],...
                  'field_patters',[],...
                  'bias',[]);
T = size(dat.x, 1);
n_channels = size(dat.x,2);
n_epos = size(dat.x,3);

%% train ICA
if isempty(opt.bias) || isempty(opt.filters)
    % do pca for whitening
    [pca_dat, pca_opt] = proc_pca(dat, 'whitening', 1);
    opt.bias = pca_opt.bias;
    
    % from here on we are in pca space
    
    if ndims(pca_dat.x) > 2
        % concatenate all epos
        pca_dat = epoToCnt(pca_dat);
    end
    % call the jade function
    [foo, P, b] = proc_pr_jade(pca_dat, n_channels, 'policy', 'number_of_features');
    % project the ica filters and field patterns from pca space to sensor space
    opt.filters = pca_opt.filters * P;
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

