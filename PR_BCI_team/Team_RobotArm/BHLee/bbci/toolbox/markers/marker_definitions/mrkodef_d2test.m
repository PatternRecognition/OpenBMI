function mrk= mrkodef_d2test(mrko, opt)

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'response_markers', {'R 16', 'R  8'});


respDef= cat(1, fliplr(opt.response_markers), {'left', 'right'});

%class 'both' is not handled by mrk_matchStimWithResp
%respDef= {'R  8', 'R 16', 'R 24';
%          'left', 'right', 'both'};

target_stim_codes= [135 139 141 147 149 153];
stimDef= {target_stim_codes, setdiff(128:159, target_stim_codes);
          'target','non-target'};
miscDef= {100, 101, 102, 252, 253;
          'cue off', 'response time', 'time off', 'start', 'end'};
mrk_stim= mrk_defineClasses(mrko, stimDef);
mrk_resp= mrk_defineClasses(mrko, respDef);
mrk_misc= mrk_defineClasses(mrko, miscDef);

[mrk, istim, iresp]= ...
    mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                          'missingresponse_policy', 'accept', ...
                          'multiresponse_policy', 'first', ...
                          'removevoidclasses', 0);
ivalid= find(~mrk.missingresponse);
ishit= any(mrk.y(:,ivalid)==mrk_resp.y([1 2],iresp));
mrk.ishit= NaN*ones(size(mrk.pos));
mrk.ishit(ivalid)= ishit;
mrk= mrk_addIndexedField(mrk, 'ishit');
mrk.misc= mrk_misc;
