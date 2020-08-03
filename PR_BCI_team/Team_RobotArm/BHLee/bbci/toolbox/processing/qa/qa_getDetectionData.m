function R = qa_getDetectionData(dat, stimlevs)
%
% USAGE:       function R = qa_getDetectionData(dat, stimlevs)
%
% IN:       dat         -       a epo or marker structure
%           stimlevs    -       vector of psychophysical stimulus levels: 
%                               e.g. stimlevs = [1 3 5 6 7 10];
%
% OUT:      R           -       matrix of size #StimulusLevels x 3:
%                               1st column: Stimulus Level
%                               2nd column: Proportion of stimuli detected
%                                           in this stimulus level
%                               3rd column: #Trials in Stimulus Level
%
% Simon Scholler, June 2011
%

nStimlev = length(dat.className);
if nargin<2
    if isfield(dat,'stimlevel')
        stimlevs = unique(dat.stimlevel);
    else
        stimlevs = 1:nStimlev;
    end
end

R = zeros(nStimlev,3);
for sl = 1:nStimlev
    det_rate = dat.detected(logical(dat.y(sl,:)));
    R(sl,:) = [stimlevs(sl) mean(det_rate) length(det_rate)];
end
