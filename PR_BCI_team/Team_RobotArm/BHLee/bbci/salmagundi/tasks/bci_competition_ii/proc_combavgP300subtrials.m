function epo_ave= proc_combavgP300subtrials(epo, Nk, policy)
%epo= proc_combavgP300subtrial(epo, Nk, <policy='nchoosek'>)
%
% IN  policy - nchoosek: Nk= [N k], all k-element subsets of 1:N
%              cyclic:   {[1..N], [1+k..N+k], ...}

if ~isfield(epo,'code'),
  error('epo must contain a field ''code'' (copied from mrk)');
end

nBlocksPerSel= 15;
nTrialsPerBlock= 12;
if ~exist('Nk','var'), Nk=[nBlocksPerSel 1]; end
if ~exist('policy','var'), policy= 'nchoosek'; end

N= Nk(1);
k= Nk(2);
switch(policy),
 case 'nchoosek',
  idx_blox= nchoosek(1:N, k);
 case 'cyclic',
  step= Nk(3);
  nnt= ceil(N/step);
  idx_blox= zeros(nnt, k);
  for mt= 1:nnt,
    idx_blox(mt,:)= [1:k] + step*(mt-1);
  end
  idx_blox= mod(idx_blox-1, N)+1;
 otherwise,
  error('policy not known');
end
nNewTrialsPerSel= size(idx_blox,1);

  
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
    for nn= idx_blox(mt,:),
      src= (is-1)*SB + (nn-1)*nTrialsPerBlock;
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
epo_ave.x= epo_ave.x / size(idx_blox,2);
