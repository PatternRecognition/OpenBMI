function epo_ave= proc_averageP300subtrials(epo, N, maxTrialsPerSel)
%epo= proc_albanyAverageP300Trials(epo, <N, maxTrialsPerSel>)
%
% in the albany P300 data set for each letter selection there are
% 15 blocks of 12 trials. in each block each column (code 1-6)
% and each row (code 7-12) is intensified exactly once.
% this function averages the trials of N blocks with matching
% codes. the first new trial is the average of those trials in
% the first N blocks of the first selection that have code 1,
% the second new trial is the average of those trials in 
% the first N blocks of the first selection that have code 2,
% and so on. the 13th new trial is the average of those trials in
% the first N blocks of the second selection that have code 1, ...
%
% confused? never mind, i am not sure if this works anyway.

if ~isfield(epo,'code'),
  error('epo must contain a field ''code'' (copied from mrk)');
end

nBlocksPerSel= 15;
if ~exist('N','var'), N=nBlocksPerSel; end
if ~exist('maxTrialsPerSel','var'), maxTrialsPerSel=1; end

if length(N)>1,
  if multipleTrialsPerSel, error('invalid input format'); end
  idx_blox= N;
  N= length(idx_blox);
else
  idx_blox= 1:N;
end

nNewTrialsPerSel= min(floor(nBlocksPerSel/N), maxTrialsPerSel);
if nNewTrialsPerSel<maxTrialsPerSel,
  warning(sprintf('only %d trials can be calculted per selection', ...
                  nNewTrialsPerSel));
end

nTrialsPerBlock= 12;
[T, nChans, nTrials]= size(epo.x);
nNewTrials= nTrials / nBlocksPerSel * nNewTrialsPerSel;
epo_ave= copyStruct(epo, 'x','y');
epo_ave.x= zeros([T, nChans, nNewTrials]);
if isfield(epo, 'y'),
  epo_ave.y= zeros(2, nNewTrials);
end
epo_ave.code= zeros(1, nNewTrials);
epo_ave.base= zeros(1, nNewTrials);

SB= nTrialsPerBlock*nBlocksPerSel;
TiB= 1:nTrialsPerBlock;
iv_trg= TiB;
for is= 1:nTrials/SB,
  for mt= 1:nNewTrialsPerSel,
    for nn= idx_blox,
      src= (is-1)*SB + (mt-1)*N*nTrialsPerBlock + (nn-1)*nTrialsPerBlock;
      [so,si]= sort(epo.code(src + TiB));
      epo_ave.x(:,:,iv_trg)= epo_ave.x(:,:,iv_trg) + ...
                             epo.x(:,:,src+si);
    end
    if isfield(epo, 'y'),
      epo_ave.y(:,iv_trg)= epo.y(:,src+si);
    end
    epo_ave.code(iv_trg)= TiB;
    epo_ave.base(iv_trg)= is;
    iv_trg= iv_trg + nTrialsPerBlock;
  end
end
epo_ave.x= epo_ave.x / N;
