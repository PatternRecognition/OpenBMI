function mrk= mrkodef_imag_fbarrow(mrko, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'classes_fb', {'target 1','target 2'}, ...
                  'stim_desc', [1 2]);

                
stimDef= {opt.stim_desc(1), opt.stim_desc(2);
          opt.classes_fb{:}};
respDef= {11, 12, 21, 22, 23, 24, 25, 29;
          ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
          ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}], ...
          'timeout', 'timeout', 'timeout', 'end-of-trial'};
miscDef= {60, 70, 71, 41, 42, 51, 52;
          'free', 'rotation', 'rotation off', ...
          'touch left correct', 'touch right correct', ...
          'touch left false', 'tough right false'};

opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef);

mrk_stim= mrk_defineClasses(mrko, opt.stimDef);
mrk_resp= mrk_defineClasses(mrko, opt.respDef);
mrk_misc= mrk_defineClasses(mrko, opt.miscDef);

%mrk_intro= mrk_chooseEvents(mrk_stim, 1:opt.adapt_trials);
%mrk_stim= mrk_chooseEvents(mrk_stim, opt.adapt_trials+1:length(mrk_stim.pos));
[mrk, istim, iresp]= mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                                           'multiresponse_policy', 'first', ...
                                           'removevoidclasses', 0);
mrk.duration= mrk.latency;
mrk= rmfield(mrk, 'latency');
idxcl= getClassIndices(mrk_resp, 'hit*');
mrk.ishit= any(mrk_resp.y(idxcl,iresp),1);
idxcl= getClassIndices(mrk_resp, 'timeout');
idxcl= union(idxcl, getClassIndices(mrk_resp, 'reject*'));
mrk.isreject= any(mrk_resp.y(idxcl,iresp),1);
mrk.indexedByEpochs= {'duration', 'resp_toe', 'ishit','isreject'};
mrk.misc= mrk_misc;

%mrk.intro= ...
%    mrk_matchStimWithResp(mrk_intro, ...
%                          mrk_chooseEvents(mrk_resp, 1:opt.adapt_trials), ...
%                          'removevoidclasses', 0);
%mrk.intro.duration= mrk.intro.latency;
%mrk.intro= rmfield(mrk.intro, 'latency');
%mrk.intro.ishit= ismember(mrk.intro.resp_toe, [11 12]);
%mrk.intro.indexedByEpochs= {'duration', 'resp_toe', 'ishit'};
