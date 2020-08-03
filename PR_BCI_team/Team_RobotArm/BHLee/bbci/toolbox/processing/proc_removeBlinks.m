function [cnt,values] = proc_removeBlinks(cnt, cntarti, mrkarti, options)
%[cnt,values] = proc_removeBlinks(cnt, cntarti, mrkarti, options)
%
% cnt is the real EEG
% cntarti are the EEG of the artifact measurement
% mrkarti are the markes in the artifact measurement
% options is a struct with the following entries
%        channels: a cell array with channels in the usual format
%        for make_Segments, where the algorithm only makes PCA. You
%        get then only this channels as ocular potentials back
%        (default all)
%        name: the name of the artifact (default: 'blinzeln')
%        method: a method to find the blinks (default:
%        readSamples_blinks) (first argument: cntarti, second
%        argument: intervals, range will be given as field in the
%        third argument
%        methodoptions: options for the call of the method (third
%        argument) (default [], readSamples_blinks has senseful
%        default (see there for documnetation))
%        lowpass: a frequenzband for lowpassfiltering (of the
%        artifcat measurement (default: no lowpass)
%        number: number of SourceVectors you want to see for
%        (default: 1)
%        range: a range around the meanpoint of the blink (default:
%        [-300 300] 
% options can be a vector (2-dim) and then it is the range
%        
% Output: fv, the corrected EEG
%         values struct with the following entries
%         variance: proportion of the variance 
%         SV: the SOurceVectors as matrice
%         SW: the SourceWaveforms as matrice
%
% Guido Dornhege
% 03.04.02

% default
defaultrange = [-300 300];
defaultchannel = [];
defaultname = 'blinzeln';
defaultmethod = 'readSamples_blinks';
defaultmethodoptions = [];
defaultlowpass = [];
defaultnumber = 1;

% check input
if nargin<3
  error('not enough input arguments');
end

if ~exist('options') | isempty(options)
  options = defaultrange;
end

if ~isstruct(options)
  options.range = options;
end

if ~isfield(options,'range')
  options.range = defaultrange;
end

if ~isfield(options,'channels')
  options.channels = defaultchannel;
end

if ~isfield(options,'name')
  options.name = defaultname;
end

if ~isfield(options,'method')
  options.method = defaultmethod ;
end

if ~isfield(options,'methodoptions')
  options.methodoptions = defaultmethodoptions;
end

if ~isfield(options,'lowpass')
  options.lowpass = defaultlowpass ;
end

if ~isfield(options,'number')
  options.number =defaultnumber;
end

% lowpass
if ~isempty(options.lowpass)
  cntarti = proc_filtByFFT(cntarti,options.lowpass);
end


% now prepare the call of get_ArtifactsPCA

opt.same = 0;
opt.channels = options.channels;
blinzeln.name = options.name;
blinzeln.artifacts = options.name;
blinzeln.number = options.number;
blinzeln.readSamples = options.method;
blinzeln.readOptions = options.methodoptions;
blinzeln.readOptions.range = options.range;


% now get the values
if nargin<2
    arti  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, blinzeln);
else
    [arti,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, blinzeln);
    values.SW = value.SourceWaveforms{1};
    values.variance = value.values;
    values.SV = value.SourceVectors;
end

if ~isempty(options.channels)
  cnt = proc_selectChannels(cnt, options.channels{:});
end

cnt.x = cnt.x - arti.x;
