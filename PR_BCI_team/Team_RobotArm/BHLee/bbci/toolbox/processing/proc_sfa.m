function [dat, opt] = proc_sfa(dat, varargin)
% [dat_sfa, sfa_opt] = proc_sfa(dat, <opt>)
% 
% Linear Slow Feature Analysis (e.g. Wiskott 2002, Berkes and Wiskott 2005)
% Can be applied to continuous or epoched data. 
%
% Train SFA:
% [dat_sfa_train, sfa_opt] = proc_sfa(dat_train)
% 
% Apply SFA:
% dat_sfa_test = proc_sfa(dat_test, sfa_opt)
%
% sfa_opt contains a bias vector as well as filters and field patterns of the 
% sources.
%
% 
% Sven Daehne, 03.2011, sven.daehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'filters', [] ...
    ,'field_patterns', [] ...
    ,'bias', [] ...
);

if isempty(opt.filters) || isempty(opt.bias)
    %% train SFA
    [foo, W, foo, b, sfa1_opt] = sfa1(dat.x);
    opt.filters = W;
    opt.field_patterns = inv(opt.filters)';
    opt.bias = b;
    opt.C = sfa1_opt.C;
    opt.C_dot = sfa1_opt.C_dot;
    
end

%% apply SFA
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


