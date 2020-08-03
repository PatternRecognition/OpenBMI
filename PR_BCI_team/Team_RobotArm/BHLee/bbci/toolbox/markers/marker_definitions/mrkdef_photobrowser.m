function mrk= mrkodef_photobrowser(mrko, varargin),


stimDef= {[20:32], [120:132]; ...
          'Non-target', 'Target'};
trialDef= {1, 2; ...
          'start', 'stop'};
      
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'stimDef', stimDef, ...
                  'nClasses', 12, ...
                  'nRepetitions', 12, ...
                  'trialDef', trialDef);
              
[tOff tid] = max([opt.stimDef{1,1}(1), opt.stimDef{1,2}(1)]);
[ntOff ntid] = min([opt.stimDef{1,1}(1), opt.stimDef{1,2}(1)]);
              
mrk = mrk_defineClasses(mrko, opt.stimDef);
tr_mrk = mrk_defineClasses(mrko, opt.trialDef);

mrk.trial_idx = zeros(1,length(mrk.pos));
mrk.block_idx = mrk.trial_idx;
mrk.idx_in_block = mrk.trial_idx;

trstarts = mrk_selectEvents(tr_mrk, tr_mrk.toe == 1);
nrTrials = length(trstarts.pos);
for trnum = 1:nrTrials,
    if trnum < nrTrials,
        intr = mrk.pos > trstarts.pos(trnum) & mrk.pos < trstarts.pos(trnum+1);
    else
        intr = mrk.pos > trstarts.pos(trnum);
    end
    nrintr = length(find(intr));
    clintr = nrintr / (opt.nRepetitions /2);
    mrk.trial_idx(intr) = trnum;
    mrk.block_idx(intr) = reshape(repmat([1:opt.nRepetitions/2], clintr, 1), 1, []);
    mrk.idx_in_block(intr) = reshape(repmat([1:opt.nRepetitions]', 1, (opt.nRepetitions /2)), 1, []);
end

stimulus = mrk.toe;
stimulus(ismember(mrk.toe, opt.stimDef{1,tid})) = stimulus(ismember(mrk.toe, opt.stimDef{1,tid})) - (tOff - ntOff);
mrk.stimulus= stimulus-ntOff+1;
mrk= mrk_addIndexedField(mrk, 'stimulus');
              