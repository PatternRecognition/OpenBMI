function [dat_csp, CSP_W, CSP_D] = func_csp(dat, varargin)
% func_csp:
%     Computes common spatial patterns (CSP) mehtods.
% 	  this version supports binary classes. Multi class csp version will be
% 	  updated soon.
%
% Example:
%     [DAT_CSP, CSP_W, CSP_D] = func_csp(SMT, {'n_patterns', [3]});
%     [DAT_CSP, CSP_W, CSP_D] = func_csp(SMT, {'n_patterns', [3]; 'cov', 'average'});
%
% Input:
%     dat - Data structure of epoched
%
% Options:
%     n_patterns - number of patterns
%     cov        - 'normal' or 'average'
%     score      - 'eigenvalue'
%     policy     - 'normal' or 'directors_cut'
%
% Returns:
%     dat_csp  - Obtained data calculated by CSP function.
%     CSP-W    - Values of CSP weight.
%     CSP_D    - Values of CSP score.

if nargin == 0
    error('OpenBMI: Input data should be specified');
end

if ~all(isfield(dat, {'x', 'y_logic', 'class', 'chan'}))
   error('OpenBMI: Data must have fields named ''x'', ''y_logic'', ''class'', and ''chan''');
end

if ndims(dat.x) ~= 3
    error('OpenBMI: Input data should be epoched');
end

% Default parameters for CSP
opt_default = {'n_patterns', 3;
    'cov', 'normal';
    'score', 'eigenvalue';
    'policy', 'normal'};

if isempty(varargin)  % Default parameter setting
    opt = opt_cellToStruct(opt_default);
else
    if iscell(varargin{:})
        opt = opt_cellToStruct(varargin{:});
    elseif isstruct(varargin{:}) % already structure(x-validation)
        opt = varargin{:};
    end
    for i = 1:length(opt_default) % setting default parameters if some of input parameter is missing in varargin
        if ~isfield(opt, opt_default(i, 1))
            opt.(opt_default{i, 1}) = opt_default{i, 2};
        end
    end
end

if opt.n_patterns > length(dat.chan)
    warning('OpenBMI: Check the number of patterns');
    opt.n_patterns = length(dat.chan);
end

% default setting
dat_csp = dat;
dat_csp.x = [];

% calculate classwise covariance matrices
[n_dat, ~, n_chans] = size(dat.x);
n_classes = size(dat.class, 1);
R = zeros(n_chans, n_chans, n_classes);

switch opt.cov
    case 'normal'
        for i = 1:n_classes
            idx = dat.y_logic(i, :);  %% ??? mrk.y·Î ¼öÁ¤
            %t_dat = zeros([n_dat length(idx) n_chans]);
            t_dat = dat.x(:, idx, :);
            t_dat = reshape(t_dat, [n_dat * sum(idx), n_chans]);
            R(:, :, i) = cov(t_dat);
        end
%     case 'average' %%malfunction
%         for c = 1:n_classes
%             C = zeros(n_chans, n_chans);
%             idx = find(epo.y_logical(c, :));
%             for m = idx
%                 C = C + cov(squeeze(epo.x(:, m, :)));
%             end
%             R(:, :, c) = C / length(idx);
%         end
    otherwise
        warning('OpenBMI: Check the cov options');
        for i = 1:n_classes
            idx = dat.y_logic(i, :);
            t_dat = dat.x(:, idx, :);
            t_dat = reshape(t_dat, [n_dat * sum(idx), n_chans]);
            R(:, :, i) = cov(t_dat);
        end
end
% R(isnan(R))=0;
[W, D] = eig(R(:, :, 2), R(:, :, 1) + R(:, :, 2));

switch opt.score
    case 'eigenvalue'
        score = diag(D);
    otherwise
        warning('OpenBMI: Check the score options');
        score = diag(D);
end

n_pattern = opt.n_patterns;
switch opt.policy
    case 'normal'
        CSP_W = W(:, [1:n_pattern, end - n_pattern + 1:end]);
        CSP_D = score([1:n_pattern, end - n_pattern + 1:end]);
    case 'directors_cut'
        absscore = 2 * (max(score, 1 - score) - 0.5);
        [~, di] = sort(score);
        Nh = floor(n_chans / 2);
        iC1 = find(ismember(di, 1:Nh, 'legacy'));
        iC2 = flipud(find(ismember(di, n_chans - Nh + 1:n_chans, 'legacy')));
        iCut = find(absscore(di) >= 0.66 * max(absscore));
        idx1 = [iC1(1); intersect(iC1(2:Nh - 1), iCut, 'legacy')];
        idx2 = [iC2(1); intersect(iC2(2:Nh - 1), iCut, 'legacy')];
        fi = di([idx1; flipud(idx2)]);
        CSP_W = W(:, fi);
        CSP_D = score(fi);
    otherwise
        warning('OpenBMI: Check the policy options');
        CSP_W = W(:, [1:n_pattern, end - n_pattern + 1:end]);
        CSP_D = score([1:n_pattern, end - n_pattern + 1:end]);
end

dat_csp = func_projection(dat, CSP_W);
dat_csp.x = dat.x;

dat_csp = opt_history(dat_csp, mfilename, opt);

end