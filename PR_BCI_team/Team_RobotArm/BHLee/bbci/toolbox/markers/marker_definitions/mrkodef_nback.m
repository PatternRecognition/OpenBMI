function mrk= mrkodef_nback(mrko, varargin)

stimDef= {[10:29], [30:49];
          'nomatch', 'match'};
respDef= {'S  8', 'S  7';       
          'true', 'false'};
miscDef= {'S200', 'S201',  'S252',  'S253';                               
          'contdown start', 'countdown stop', 'start', 'end'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef, ...
                       'N', 0);

mrk_stim= mrk_defineClasses(mrko, opt.stimDef);
mrk_resp= mrk_defineClasses(mrko, opt.respDef);
is= strmatch('New Segment', mrko.type);
idx_intro= [];
for ii= 1:length(is),
  seg_start= mrko.pos(is(ii));
  mrk_start= min(find(mrk_stim.pos>seg_start));
  idx_intro= [idx_intro, mrk_start + [0:opt.N-1]];
end
mrk.intro= mrk_chooseEvents(mrk_stim, idx_intro);
mrk_stim= mrk_chooseEvents(mrk_stim, 'not',idx_intro);

[mrk, istim, iresp]= ...                       
      mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                            'missingresponse_policy', 'accept', ...
                            'multiresponse_policy', 'first', ...   
                            'removevoidclasses', 0);
mrk.resp= mrk_resp;
idx= find(~mrk.missingresponse);
mrk.ishit(idx)= mrk.resp.y(1,iresp);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);                    
