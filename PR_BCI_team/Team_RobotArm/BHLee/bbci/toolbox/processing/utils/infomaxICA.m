function [Y, W, lambda, b, opt] = infomaxICA(X, varargin)
% infomax ICA
% [YYY, W, d, b, opt] = infomaxICA(X, varargin)
%
% Input:
% X - data matrix, dimensions of X are [n_samples, n_channels, <n_epos>]
%
% Output:
% YYY - unmixed sources output signal
% W - matrix containg the ICA filters in the columns
% lambda - 
% b - bias vector
% opt - contains maxiter, eta, eta_decay, thres_grad, gradient,
%       gradient_in, gradient_out, entropy
%
% adapted from
% http://jim-stone.staff.shef.ac.uk/bookICA2004/ica_appD_demo.m
%
% Johannes Hoehne 10.2011, j.hoehne@tu-berlin.de

warning('this func is still under development and not really tested, so be careful with interpretation!!!')

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'b', [], ...
    'maxiter', 100, ...
    'eta', 1, ...
    'eta_decay', 1, ... % each iteration, eta = eta * opt.eta_decay
    'thres_grad', power(10,-2), ... %if norm(grad(:)) < opt.thres_grad, then stop iterating
    'verbose', 1);

n_epos = size(X,3);
n_samples = size(X,1);
n_channels = size(X,2);
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
% zero mean for each channel (column in X)
X = X - repmat(b, size(X,1), 1);


% Initialise unmixing matrix W to identity matrix.
W = eye(n_channels,n_channels);

% Initialise Y, the estimated source signals.
Y = X*W;


% Make array hs to store values of function and gradient magnitude.
hs=zeros(opt.maxiter,1);
gs=zeros(opt.maxiter,1);
curr_eta = opt.eta;

% Begin gradient ascent on h ...
for iter=1:opt.maxiter
    if opt.verbose && mod(iter, ceil(opt.maxiter/25)) == 0
        fprintf('iteration %d/%d\n', iter, opt.maxiter);
    end
    % Get estimated source signals, Y.
    Y = X*W; % wt vec in col of W.
    % Get estimated maximum entropy signals YYY=cdf(Y).
    YYY = tanh(Y);
    % Find value of function h.
    % h = log(abs(det(W))) + sum( log(eps+1-YYY(:).^2) )/n_samples;
    detW = abs(det(W));
    h = ( (1/n_samples)*sum(sum(YYY)) + 0.5*log(detW) );
    % Find matrix of gradients @h/@W_ji ...
    g = inv(W') - (2/n_samples)*X'*YYY;
    % Update W to increase h ...
    W = W + curr_eta*g;
    % Record h and magnitude of gradient ...
    magn_grad = norm(g(:));
    hs(iter)=h; gs(iter)=magn_grad;
    if magn_grad < opt.thres_grad
        break
    end
    curr_eta = curr_eta * opt.eta_decay;
    if iter > 4 
        if hs(iter) <  hs(iter-1) && hs(iter-1) >  hs(iter-2) && hs(iter-2) <  hs(iter-3)
             warning('infomaxICA is jumping over the minimum, reducing eta!');
             curr_eta = curr_eta * 0.2;
        end
    end
end;

if iter==opt.maxiter
    warning('infomaxICA stopped because the maximium number of iterations (%i) was exceeded! Last norm(gradient) = %05d, stopping threshold was set to %05d!' ,opt.maxiter, magn_grad, opt.thres_grad)
end

vars = var(Y,1);
lambda = vars / sum(vars);

opt.entropy_out = hs(iter);
opt.entropy_in = hs(1);
opt.gradient = gs(1:iter);
opt.entropy = hs(1:iter);


if is_epo
    % bring Y in the same structure that X had
    Y = reshape(Y, [n_samples+1, n_epos, n_channels]);
    Y = permute(Y,[1,3,2]); % dims are now: [time, channels, epo]
    Y = Y(1:(end-1),:,:); % remove the zero-sample 
end

return
