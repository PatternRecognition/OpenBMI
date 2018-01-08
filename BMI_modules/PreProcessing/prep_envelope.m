function [out] = prep_envelope(dat,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_envelope
%
% Synopsis:
%   [out] = prep_envelope(dat,<var>)
%
% Example :
%    [out] = prep_envelope(dat)
%
% Arguments:
%     dat    - Epoched signal
% Options:
%     Time[ms] - time window. scalar or nx1 vector for weighting (default: 100)
%     Method - 'centered' or 'causal' (default: causal)
%
% Returns:
%     out - Envelope of the signal
%
% Description:
%     This function smoothly outlines the extremes of an oscillating
%     signal, continuous or epoched.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 01-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end

s = size(dat.x);
dat.x= reshape(abs(hilbert(dat.x(:,:))),s);
out= prep_movingAverage(dat,varargin{:});

