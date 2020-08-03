function mrk_jit = mrk_addResponseLatency(mrk, stimresp, varargin)
%mrk_addResponseLatency
% add a field 'latency' [ms] for every stimulus-response pair in stimresp.
%
% USAGE: mrk_jit= addResponseJitter(mrk, stimresp, max_latency)
%
% IN:  mrk           struct containing markers (pos, toe, fs)
%      stimresp      cell array ; 
%                    each row indicates class of stimuli and a class
%                    of correct responses
%      max_latency   double array indicating the maximal latency in ms
%                    (1: left lat(ATTENTION: positive) , 2: right lat)
% OUT: mrk_jit       struct containing class markers (pos, toe, fs, latency)

mrk_jit= mrk;
lat= NaN*ones(size(mrk.pos));
for ii= 1:size(stimresp,1),
  mrk.toe= NaN*ones(size(mrk.pos));
  [dmy, stim]= mrk_selectClasses(mrk, stimresp{ii,1});
  [dmy, resp]= mrk_selectClasses(mrk, stimresp{ii,2});
  mrk.toe(stim)= 1;
  mrk.toe(resp)= -1;
  mrk= getResponseJitter(mrk, {1, -1}, varargin{:});
  valid= find(~isnan(mrk.latency));
  lat(valid)= mrk.latency(valid);
end
mrk_jit.latency= lat;
mrk_jit.indexedByEpochs= mrk.indexedByEpochs;
