function [dat, opt] = proc_ica_infomax(dat, varargin)
% [dat_ica, ica_opt] = proc_ica(dat, <opt>)
% 
% infomax Independent Component Analysis (Bell-Sejnowski)
% Can be applied to continuous or epoched data. 
%
% Train ICA:
% [dat_ica_train, ica_opt] = proc_ica(dat_train)
% 
% Apply ICA:
% dat_ica_test = proc_ica(dat_test, ica_opt)
%
% ica_opt contains a bias vector as well as filters and field patterns of the 
% sources.
%
% Johannes Hoehne 10/2011

warning('this func is still under development and not really tested, so be careful with interpretation!!!')

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'filters', [] ...
    ,'field_patterns', [] ...
    ,'bias', [] ...
);

if isempty(opt.filters) || isempty(opt.bias)
    %% train ICA
    [foo, W, lambda, b, ica_opt] = infomaxICA(dat.x);
    opt.filters = W;
    opt.bias = b;
    opt.field_patterns = inv(opt.filters)';    
end

%% apply ica
n_epos = size(dat.x, 3);
n_channels = size(dat.x, 2);
T = size(dat.x, 1);
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

return
