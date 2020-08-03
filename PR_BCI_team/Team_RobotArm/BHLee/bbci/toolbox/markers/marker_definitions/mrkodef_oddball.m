function mrk= mrkodef_oddball(mrko, varargin)

stimDef= {'S  2', 'S  1';
          'dev','std'};
respDef= {'R 16', 'R  8', 'R 24',
          'left', 'right', 'both'};
miscDef= {100, 252, 253;
          'cue off', 'start', 'end'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef);

mrk_stim= mrk_defineClasses(mrko, opt.stimDef);
mrk_resp= mrk_defineClasses(mrko, opt.respDef);
mrk_misc= mrk_defineClasses(mrko, opt.miscDef);

[mrk, istim, iresp]= ...
    mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                          'missingresponse_policy', 'accept', ...
                          'multiresponse_policy', 'first', ...
                          'removevoidclasses', 0);
ivalid= find(~mrk.missingresponse);
mrk.ishit= any(mrk.y(:,ivalid)==mrk_resp.y([1 2],iresp));
mrk= mrk_addIndexedField(mrk, 'ishit');
mrk.misc= mrk_misc;
