function [cnt,SV,SW,portion] = proc_MSEC(cnt, cnt_cal, mrk_cal, window, numEEGSources, numEyeSources, opt)
% [cnt,SourceVector, SourceWaveforms, portion]= ...
%    proc_MSEC(cnt, cnt_cal, mrk_cal, <window = [-500 500], ...
%              numEEGSources=12, numEyeSources=3, opt.mean=1, opt.base=0>)
%
% INPUT:
% cnt is the usual cnt, or epo struct
% cnt_cal is the calibration data set
% mrk_cal are the markers of relevant activation (array)
% window is an array which describes the window around the peak
% numEEGSources is the number of Sources in the brain 
% numEyeSources is the number of eye Sources
% opt.mean is a flag, if it is true the mean of all features is determined before calculating SourceVectors, else on all intervals SourceVectors are determined.
% opt.base is zero and nothing happened or is an interval which describes the baseline interval regarding to each marker
%
% OUTPUT:
% cnt = the corrected Waveform
% SV = the SourceVectors
% SW = the SourceWaveforms (in a row)
% portion = the portion how many variance is declared by the eye forms in the calibration data set
%
% Guido Dornhege
% 14.08.2002

% setting defaults

if nargin<3
     error('not enough input arguments');
end

if ~exist('window') | isempty(window)
     window = [-500 500];
end

if ~exist('numEEGSources') | isempty(numEEGSources)
     numEEGSources = 12;
end

if ~exist('numEyeSources') | isempty(numEyeSources)
     numEyeSources = 3;
end

if ~exist('opt') | isempty(opt)
     opt.mean = 1;
end

if ~isfield(opt,'mean')
     opt.mean = 1;
end

if ~isfield(opt,'base')
     opt.base = 0;
end

% calculate the SourceVectors

window = round(cnt.fs*window/1000);
cali = [];
for i = mrk_cal
cali = cat(3,cali,cnt_cal.x((window(1):window(2))+i,:));
end

if length(opt.base)>1
     basi = [];
     for i = mrk_cal
          basi = cat(3,basi,cnt_cal.x((opt.base(1):opt.base(2))+i,:));
     end
     basi = mean(basi,1);
     cali = cali - repmat(basi,[size(cali,1),1,1]);
     clear basi;
end

if opt.mean
     cali = mean(cali,3);
else
     calim = mean(cali,3);
     cali = permute(cali,[2 1 3]);
     cali = reshape(cali,[size(cali,1), size(cali,2)*size(cali,3)]);
     cali = cali';
end

co = cali'*cali/(size(cali,1));

[SV,D] = eigs(co,numEEGSources+numEyeSources, 'LM',struct('disp',0));

PSV = pinv(SV);

if nargout>2
      if ~opt.mean
          cali = calim;
          clear calim
      end
      SW = PSV*cali';
      if nargout>3
           D = diag(D);
           portion = sum(D(1:numEyeSources))/sum(D);
      end
end

SVeye = SV(:,1:numEyeSources);
for i = 1:size(cnt.x,3)
    dat = cnt.x(:,:,i);
    dateye = PSV*dat';
    dateye = dateye(1:numEyeSources,:);
    dateye = SVeye*dateye;
    cnt.x(:,:,i) = dat-dateye';
end
