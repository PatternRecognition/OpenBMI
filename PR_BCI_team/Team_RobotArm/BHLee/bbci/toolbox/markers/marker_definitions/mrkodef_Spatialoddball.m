function mrk= mrkodef_Spatialoddball(mrko, opt)

if opt.individual,
    stimDef= {1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18;
             'st1','st2','st3','st4','st5','st6','st7','st8','tar1','tar2','tar3','tar4','tar5','tar6','tar7','tar8'};
    respDef= {'R  1',
              'response'};
    miscDef= {'S100', 'S252', 'S253';
              'cue off', 'start', 'end'};         
else
    target_stim_codes= [1:8];
    non_target_stim_codes= [11:18];

    stimDef= {non_target_stim_codes, target_stim_codes;
              'Target','Non-target'};
    if isfield(opt, 'countMarkers'), 
        respDef = {opt.countMarkers; 
                    'response'};
    else
        respDef= {'R  1',
                  'response'};
    end
    miscDef= {252, 253;
              'start', 'end'};
end

mrk_stim= mrk_defineClasses(mrko, stimDef);
mrk_resp= mrk_defineClasses(mrko, respDef);
mrk_misc= mrk_defineClasses(mrko, miscDef);

if length(mrk_resp.pos) > 0,
    [mrk, istim, iresp]= ...
        mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                          'missingresponse_policy', 'accept', ...
                          'multiresponse_policy', 'first', ...
                          'removevoidclasses', 0);
        ivalid= find(~mrk.missingresponse);
        hits = intersect(find(mrk.toe >10), ivalid);
        mrk.ishit = zeros(1,length(mrk.toe));
        mrk.ishit(hits) = 1;
        mrk= mrk_addIndexedField(mrk, 'ishit');
        mrk.misc= mrk_misc;
else
    mrk = mrk_stim;
end

mrk.misc= mrk_misc;

