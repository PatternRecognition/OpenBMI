function [out] = prep_resample(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_RESAMPLE - changes the sampling rate of the given EEG signal
% prep_resample (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_resample(DAT, fs, <OPT>)
%
% Example :
%     [out] = prep_resample(dat, {'fs',sampling_rate, 'n_remove', n_samples})
%     [out] = prep_resample(dat, {'fs',sampling_rate})
%     [out] = prep_resample(dat, sampling_rate, n_samples)
%     [out] = prep_resample(dat, sampling_rate)
%
% Arguments:
%     dat - Structure. Continuous data or epoched data
%         - Data which channel is to be selected     
%     fs[Hz] - Desired sampling frequency (scalar)
%     varargin - struct or property/value list of optional properties:
%          : n_remove(scalar) - Remove the first and last Nr samples from the
%            resampled dat to avoid edge effects (default is 0)
%           
% Returns:
%     out - Data structure which changed sampling freqeuncy 
%
%
% Description:
%     This function changes the sampling rate of the given EEG signal.
%     It can do both up/downsampling, considering frequency up to 3 digits.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
	warning('OpenBMI: Sampling rate should be specified')
    return
elseif isnumeric(varargin{1})%¼ýÀÚ·Î µé¾î¿È
    opt.fs = varargin{1};
    if size(varargin) == 1
        opt.n_remove = 0;
    else
        opt.n_remove = varargin{2};
    end
else %¼¿·Îµé¾î¿È
    opt = opt_cellToStruct(varargin{:});
    if ~isfield(opt,'n_remove')
        opt.n_remove = 0;
    end
end

if ~isfield(dat,'x') || ~isfield(dat,'fs')
    warning('OpenBMI: Data must have fields named ''x'',''fs''')
    return
end

fs = opt.fs;
p = round(1000*fs);
q = round(1000*dat.fs);

if ndims(dat.x) == 3
    n_t = size(dat.x,1);
    n_tr = size(dat.x,2);
    n_ch = size(dat.x,3);
    x = zeros(ceil(n_t*p/q),n_tr,n_ch);
    for i = 1:n_tr
        xt = reshape(dat.x(:,i,:),[n_t,n_ch]);
        xt = resample(xt,p,q);
        x(:,i,:) = reshape(xt,[size(xt,1),1,n_ch]);
    end
    if isfield(dat,'ival')
        out = rmfield(dat,{'x','fs','ival'});
        out.ival = linspace(dat.ival(1),dat.ival(end),size(x,1));
        out.ival = out.ival((opt.n_remove+1):end-opt.n_remove);
    end
    x = x((opt.n_remove+1):end-opt.n_remove,:,:);
elseif ismatrix(dat.x)
    out = rmfield(dat,{'x','fs'});
    x = resample(dat.x,p,q);
    x = x((opt.n_remove+1):end-opt.n_remove,:);
else
    warning('Check for the dimension of input data')
    return
end

if isfield(dat, 't')
    lag = dat.fs/fs;
    t = dat.t./lag;
    out.t = t;
end
out.fs = fs;
out.x = x;

if ~exist('opt','var')
    opt = struct([]);
end
if ~isfield(dat,'history')
    out.history = {'prep_resample',opt};
else
    out.history(end+1,:) = {'prep_resample',opt};
end
