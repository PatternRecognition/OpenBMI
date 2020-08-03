function [dat, opt] = proc_ica_sobi(dat, varargin)
% [dat, opt] = proc_ica_sobi(dat, <opt>)
% 
%  Second Order Blind Identification (SOBI) by joint diagonalization of
%          correlation  matrices
% Can be applied to continuous or epoched data. 
%
% Train ICA:
% [dat_sobi_train, sobi_opt] = proc_ica_sobi(dat_train)
% 
% Apply SFA:
% dat_sobi_test = proc_ica_sobi(dat_sobi_test, sobi_opt)
%
% sobi_opt contains 
%       filters (REQUIRED FOR EACH ICA)
%       field patterns (REQUIRED FOR EACH ICA)
%       bias vector 
%       V
%       M
%
% 
% JohannesHoehne, 12.2011, johannes.hoehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'filters', [] ...opt
    ,'field_patterns', [] ...
    ,'bias', [] ...
);

if isempty(opt.filters) || isempty(opt.bias)
    %% train SOBI
    [foo, W, V, b, sobi1_opt] = sobi1(dat.x);
    opt.filters = W;
    opt.field_patterns = sobi1_opt.winv;
    opt.bias = b;
    opt.V = V;
    opt.M = sobi1_opt.M;
end

%% apply SOBI
n_epos = size(dat.x, 3);
n_channels = size(dat.x, 2);
T = size(dat.x, 1);
if not(size(opt.bias,1) == n_channels)
    error('Dimension of bias must equal the number of channels!')
end
% make sure opt.bias is a row vector
if size(opt.bias, 1) > size(opt.bias,2)
    opt.bias = permute(opt.bias, [2 1 3]);
end

B = squeeze(repmat(opt.bias, [T,1,1]));
% subtract bias, then apply filters
dat.x = dat.x - B;
dat = proc_linearDerivation(dat, opt.filters);
