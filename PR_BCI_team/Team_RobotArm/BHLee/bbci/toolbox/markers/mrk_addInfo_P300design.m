function mrk= mrk_addInfo_P300design(mrk, nClasses, nRepetitions, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'nIntroStimuli', 0, ...
                       'nExtroStimuli', 0);

nStimuliPerTrial= opt.nIntroStimuli + nRepetitions*nClasses + ...
    opt.nExtroStimuli;
nTrials= length(mrk.pos) / nStimuliPerTrial;
if nTrials ~= floor(nTrials),
  warning('Some left-over stimuli. This should not happen.');
end
mrk.trial_idx= zeros(size(mrk.pos));
mrk.block_idx= zeros(size(mrk.pos));
mrk.idx_in_block= zeros(size(mrk.pos));
mrk= mrk_addIndexedField(mrk, {'trial_idx', 'block_idx', 'idx_in_block'});
idx_flanker= [];
extro_offset= opt.nIntroStimuli + nRepetitions*nClasses;
nTargets= [];
pp= 0;
block_idx= repmat([1:nRepetitions], nClasses, 1);
block_idx= [zeros(1, opt.nIntroStimuli), ...
            block_idx(:)', ...
            ones(1, opt.nExtroStimuli)*(nRepetitions+1)];
idx_in_block= [1:opt.nIntroStimuli, ...
               repmat([1:nClasses], 1, nRepetitions), ...
               1:opt.nExtroStimuli];
for tt= 1:nTrials,
  mrk.trial_idx(pp +[1:nStimuliPerTrial])= tt;
  mrk.block_idx(pp +[1:nStimuliPerTrial])= block_idx;
  mrk.idx_in_block(pp +[1:nStimuliPerTrial])= idx_in_block;
  idx_flanker= cat(2, idx_flanker, pp + [1:opt.nIntroStimuli]);
  idx_flanker= cat(2, idx_flanker, pp + extro_offset+[1:opt.nExtroStimuli]);
  nTargets= cat(2, nTargets, sum(mrk.y(1,pp+[1:nStimuliPerTrial])));
  pp= pp + nStimuliPerTrial;
end
mrk.flanker= mrk_chooseEvents(mrk, idx_flanker);
mrk= mrk_chooseEvents(mrk, 'not',idx_flanker);
mrk.nTargets= nTargets;
