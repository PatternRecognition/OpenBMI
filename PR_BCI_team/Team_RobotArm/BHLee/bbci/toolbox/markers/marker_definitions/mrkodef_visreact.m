function mrk=mrkodef_visreact(mrko,opt)

stimDef= {2; 'cue'};

respDef= {{'R  1','R 32','R 64'}; 'resp'};                     
miscDef= {'S  1','S  3'; 'stim','intertrial'};


mrk_stim= mrk_defineClasses(mrko, stimDef);
mrk_resp= mrk_defineClasses(mrko, respDef);
mrk_misc= mrk_defineClasses(mrko, miscDef);


[mrk, istim, iresp]= mrk_matchStimWithResp(mrk_stim, mrk_resp, ... 
                         'missingresponse_policy', 'accept', ...
                         'multiresponse_policy', 'first', ...   
                         'removevoidclasses', 0);           
mrk.y= [mrk.missingresponse; ~mrk.missingresponse];
mrk.className= {'miss', 'hit'};
