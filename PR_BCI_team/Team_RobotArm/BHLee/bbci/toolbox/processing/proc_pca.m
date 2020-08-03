function [dat, opt] = proc_pca(dat, varargin)
% [dat_pca, pca_opt] = proc_pca(dat, <opt>)
% 
% Principal Component Analysis. Can be applied to continuous or epoched
% data. 
%
% Train PCA:
% [dat_pca_train, pca_opt] = proc_pca(dat_train)
% 
% Apply PCA:
% dat_pca_test = proc_pca(dat_test, pca_opt)
%
% pca_opt contains a bias vector as well as filters and field patterns of the 
% sources.
%
% OPT: struct or property/value list of optional properties:
%     .whitening    - if 1, the output dimensions will all have unit
%                       variance (default 0)
%   
%
% see also: proc_pr_pca
% 
% Sven Daehne, 03.2011, sven.daehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'filters', [] ...
    ,'field_patterns', [] ...
    ,'bias', [] ...
    ,'whitening',0 ...
);

T = size(dat.x, 1);
n_channels = size(dat.x, 2);
n_epos = size(dat.x, 3);

if isempty(opt.filters) || isempty(opt.bias)
    %% train PCA
    % get the data matrix
    if ndims(dat.x) ==3
        % since time structure does not matter, we can simply concatenate all
        % epochs to get one big data matrix
        X = permute(dat.x, [1,3,2]); % now channels are the last dimension
        X = reshape(X, [T*n_epos, n_channels]);
    else
        X = dat.x;
    end
    % remove the mean here already
    b = mean(X,1);
    B = repmat(b, [T*n_epos, 1]);
    X = X - B;
    
    [foo, pca_struct, feat_score] = proc_pr_pca(X', n_channels, opt);
    opt.filters = pca_struct.P';
    opt.field_patterns = inv(opt.filters)';
    opt.eigenvalues = feat_score;
    opt.bias = b;
end

%% apply PCA

if not(length(opt.bias) == n_channels)
    error('Dimension of bias must equal the number of channels!')
end
% make sure opt.bias is a row vector
if size(opt.bias, 1) > size(opt.bias,2)
    opt.bias = opt.bias';
end

% subtract bias, then apply filters
B = squeeze(repmat(opt.bias, [T,1,n_epos]));
dat.x = dat.x - B;
dat = proc_linearDerivation(dat, opt.filters);

