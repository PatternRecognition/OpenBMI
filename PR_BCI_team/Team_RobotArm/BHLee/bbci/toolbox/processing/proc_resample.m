function [dat, mrk] = proc_resample(dat, target_fs, varargin)
% Resamples the field dat.x such that is has the desired sampling frequency.
% dat.t and mrk (if given) will be updated as well.
%
% proc_resample(dat, target_fs, <mrk, N>)
%
% IN:
%     dat - Structure with fields x and fs. This fields
%     will be overwritten
%     target_fs - the target sampling frequency
%
%     optional (given with keywords or as opt.xxx):
%     mrk - Structure with fields fs and pos. They will be updated
%     N - remove the first and last N samples from the resampled data to
%     avoid edge effects (default is 0)
%
% OUT:
%     dat, mrk
%
% Sven Daehne, 06-2011, sven.daehne@tu-berlin.de


% dummy mrk
mrk = [];
mrk.pos = 0;

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'N', 0 ...
    ,'mrk', mrk ...
);

N = opt.N;
mrk = opt.mrk;

p = round(target_fs*1000);
q = round(dat.fs*1000);

if ndims(dat.x)==3
    nTrials = size(dat.x,3);
    for n=1:nTrials
        X(:,:,n) = resample(dat.x(:,:,n), p, q);
    end
else
    X = resample(dat.x, p, q);
end
dat.x = X;
dat.fs = target_fs;
n_samples = size(dat.x,1);
if isfield(dat, 't')
    t = linspace(dat.t(1), dat.t(end), n_samples);
    dat.t = t; % time in ms
end
mrk.pos = round(mrk.pos * p/q);
mrk.fs = target_fs;

% remove the first and the last N samples to avoid edge effects
dat.x = dat.x((N+1):end-N, :, :);
if isfield(dat, 't')
    dat.t = dat.t((N+1):end-N);
end
mrk.pos = mrk.pos-N;