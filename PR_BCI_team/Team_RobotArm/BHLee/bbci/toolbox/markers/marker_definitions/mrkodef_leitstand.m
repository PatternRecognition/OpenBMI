function mrk= mrkodef_leitstand(mrko, varargin)

stimDef= {'S 20',   'S 21',    'S 22';
          'status', 'warning', 'alert'};
respDef= {'S 40', 'S 41', 'S 70';
          'r-stat', 'r-warm', 'r-alert'};
mathDef= {'S 50', 'S 65', 'S 66', 'S 42';
           'math_start', 'math_correct', 'math_wrong', 'receipt'};
miscDef= {'S 99',  'S252',  'S253';
          'money', 'start', 'end'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'mathDef', mathDef, ...
                       'miscDef', miscDef);

mrk_stim= mrk_defineClasses(mrko, opt.stimDef);
mrk_resp= mrk_defineClasses(mrko, opt.respDef);
[mrk, istim, iresp]= ...
      mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                            'missingresponse_policy', 'accept', ...
                            'multiresponse_policy', 'first', ...
                            'removevoidclasses', 0);
mrk.resp= mrk_resp;
mrk.math= mrk_defineClasses(mrko, opt.mathDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);

[mrk_alert, idx_ma]= mrk_selectClasses(mrk_stim, 'alert');
[mrk_ms, idx_ms]= mrk_selectClasses(mrk.math, 'math_start');
[tmp,is,ir]= mrk_matchStimWithResp(mrk_alert, mrk_ms, ...
                            'missingresponse_policy', 'accept', ...
                            'multiresponse_policy', 'first', ...
                            'removevoidclasses', 0);
noresp= find(tmp.missingresponse);
% no response to last stimulus within a run can happen 
mrk.latency_turn_to_math= NaN*zeros(1, length(mrk.pos));
mrk.latency_turn_to_math(idx_ma)= tmp.latency;

[mrk_mc, idx_mc]= mrk_selectClasses(mrk.math, 'math_correct');
[tmpc]= mrk_matchStimWithResp(mrk_alert, mrk_mc, ...
                            'missingresponse_policy', 'accept', ...
                            'multiresponse_policy', 'first', ...
                            'removevoidclasses', 0);
[mrk_mw, idx_mw]= mrk_selectClasses(mrk.math, 'math_wrong');
[tmpw]= mrk_matchStimWithResp(mrk_alert, mrk_mw, ...
                            'missingresponse_policy', 'accept', ...
                            'multiresponse_policy', 'first', ...
                            'removevoidclasses', 0);
mrk.latency_solve_math= NaN*zeros(1, length(mrk.pos));
mrk.latency_solve_math(idx_ma)= tmpc.latency-tmp.latency;
idx_wrong= find(~isnan(tmpw.latency));
mrk.latency_solve_math(idx_ma(idx_wrong))= ...
    -(tmpw.latency(idx_wrong)-tmp.latency(idx_wrong));

%mrk.latency_turn_from_math= NaN*zeros(1, length(mrk.pos));
%mrk.latency_turn_from_math(idx_ma)= tmp.latency;
