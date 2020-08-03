function [Y, W, d, b, opt] = sfa1(X, varargin)
% Linear Slow Feature Analysis (Wiskott 2002, Berkes and Wiskott 2005)
% [Y, W, d, b, opt] = sfa1(X, varargin)
%
% Input:
% X - data matrix, dimensions of X are [n_samples, n_channels, <n_epos>]
% 
% Output:
% Y - SFA output signal
% W - matrix containg the SFA filters in the columns
% d - delta values, measure of slowness
% b - bias vector
% opt - contains covariance matrix of the data (opt.C) and covariance matrix of
%       the time derivative of the data (opt.C_dot)
%
% Sven Daehne, 03.2011, sven.daehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'b',[],...
                  'C',[],...
                  'C_dot',[]);

n_samples = size(X,1);
n_channels = size(X,2);
n_epos = size(X,3);

is_epo = ndims(X) > 2;

if is_epo
    % add a zero-sample before concatenating the epos
    X = [X; zeros(1, n_channels, n_epos)];  
    % concatenate the epos
    X = permute(X,[1,3,2]); % dims are now: [time, epo, channels]
    X = reshape(X, [(n_samples+1)*n_epos, n_channels]);
end


if isempty(opt.b)
    % compute the bias, i.e. average of each dimension
    b = mean(X,1);
else
    b = opt.b;
end

% subtract the mean
B = repmat(b,[size(X,1), 1]);
X = X - B;

if isempty(opt.C)
    % compute covariance matrix
    C = cov(X);
else
    C = opt.C;
end

if isempty(opt.C_dot)
    % compute second moment matrix of the time derivative
    X_dot = diff(X, 1, 1);
    C_dot = X_dot'*X_dot ./ (n_samples*n_epos);
else
    C_dot = opt.C_dot;
end

% compute SFA
[W, D] = eig(C_dot, C);
d = diag(D);
[d sort_idx] = sort(d, 'ascend');
W = W(:,sort_idx);
% project the signal
Y = X*W;


if is_epo
    % bring Y in the same structure that X had
    Y = reshape(Y, [n_samples+1, n_epos, n_channels]);
    Y = permute(Y,[1,3,2]); % dims are now: [time, channels, epo]
    Y = Y(1:(end-1),:,:); % remove the zero-sample 
end


opt.C = C;
opt.C_dot = C_dot;
opt.b = b;

