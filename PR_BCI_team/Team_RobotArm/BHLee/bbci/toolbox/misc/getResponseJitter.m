function mrk_jit = getResponseJitter(mrk, stimresp, max_latency)
% getResponseJitter  find the latency (in ms) for every stimulus in stimresp.
%
% USAGE: mrk_jit= getResponseJitter(mrk, stimresp, max_latency)
%
% IN:  mrk           struct containing markers (pos, toe, fs)
%      stimresp      cell array ; each row indicates one stimulus and an array of responses.
%      max_latency   double array indicating the maximal latency in ms (1: left lat(ATTENTION: positive) , 2: right lat)
% OUT: mrk_jit       struct containing class markers (pos, toe, fs, latency)
%
% EXAMPLE:
% mrk =
%    pos: [272, 3191, 3216, 3292, 3317, 3392, 3416, 3492, 3518, 3591]
%    toe: [25,  252,  12,   1,    12,   1,    11,   1,    11,   1]
%     fs: 100
%
% stimresp = { [1], [-74];
%              [2], [-111,-34]}

% 25.08.03 kraulem

mrk_jit = mrk;
mrk_jit.latency = NaN*zeros(1,length(mrk_jit.pos));% by default: a latency property does not apply.

for ind1 = 1:size(stimresp,1)
  % match the stimulus to its corresponding responses
  stim_indarr = find(mrk.toe==stimresp{ind1,1});
  for ind2 = 1:length(stim_indarr)
    % for every stimulus of this type in the marker array: find nearby responses.
    near_resparr = [];
    for ind3 = 1:size(stimresp{ind1,2},2)
      % find all responses
      near_resparr = [near_resparr, find(mrk.toe == stimresp{ind1,2}(ind3) & mrk.pos >= mrk.pos(stim_indarr(ind2))-mrk.fs*max_latency(1)/1000 & mrk.pos <= mrk.pos(stim_indarr(ind2))+mrk.fs*max_latency(2)/1000)];
    end
    near_resparr = sort(near_resparr);
    for ind3=1:length(near_resparr)
      % find the nearest response and set the latency accordingly
      lat = 1000*(mrk.pos(near_resparr(ind3))-mrk.pos(stim_indarr(ind2)))/mrk.fs;
      if ind3==1 | abs(lat)<abs(mrk_jit.latency(stim_indarr(ind2)))
        mrk_jit.latency(stim_indarr(ind2))=lat;
      end
    end
  end
end

if ~isfield(mrk_jit, 'indexedByEpochs'),
  mrk_jit.indexedByEpochs= {};
end
if ~ismember('latency', mrk_jit.indexedByEpochs),
  mrk_jit.indexedByEpochs= cat(2, mrk_jit.indexedByEpochs, {'latency'});
end
