function mrk= mrkodef_imag_fbgoalkeeper(mrko, varargin)

opt= propertylist2struct(varargin{:});

stimDef= {1, 2;
          opt.classes_fb{:}};
respDef= {11, 12, 22, 21, 53;
          ['correct ' opt.classes_fb{1}], ['correct ' opt.classes_fb{2}], ...
          ['incorrect ' opt.classes_fb{1}], ...
          ['incorrect ' opt.classes_fb{2}], ...
          'missed'};

miscDef= {41, 42, 51, 52, 61, 62, 71, 72;
          'hit left', 'hit right', 'miss kl-tr', 'miss kr-tl', ...
          'too late correct left', 'too late correct right', ...
          'too late incorrect kl-tr', 'too late incorrect kr-tl'};

opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef, ...
                       'missingresponse_policy', 'accept', ...
                       'multiresponse_policy', 'first');

mrk_stim= mrk_defineClasses(mrko, opt.stimDef);
mrk_resp= mrk_defineClasses(mrko, opt.respDef);
mrk_misc= mrk_defineClasses(mrko, opt.miscDef);

[mrk, istim, iresp]= ...
    mrk_matchStimWithResp(mrk_stim, mrk_resp, opt, ...
                          'removevoidclasses', 0);

idxcl= getClassIndices(mrk_resp, 'correct*');
if strcmp(opt.missingresponse_policy,'reject'),
  mrk.ishit= any(mrk_resp.y(idxcl,iresp),1);
else
  mrk.ishit= zeros(size(mrk.pos));
  irespgiven= find(~mrk.missingresponse);
  mrk.ishit(irespgiven)= any(mrk_resp.y(idxcl,iresp),1);
end
mrk= mrk_addIndexedField(mrk, 'ishit');
mrk.misc= mrk_misc;
