function mrk= mrkodef_spatial_TNO(mrko, opt)

if opt.individual,
    stimDef= {11,12,13,14,15,16,1,2,3,4,5,6;
             'tar1','tar2','tar3','tar4','tar5','tar6','st1','st2','st3','st4','st5','st6'};         
else
    non_target_stim_codes= [1:6];
    target_stim_codes= [11:16];
    stimDef= {target_stim_codes, non_target_stim_codes;
              'Target', 'Non-target'};
end

respDef= {[80:115];
          'count'};
feedbackDef = {'S 51', 'S 52', 'S 53', 'S 54', 'S 55', 'S 56', 
               'tact1', 'tact2', 'tact3', 'tact4', 'tact5', 'tact6'};
errDef = {200, 201;
          'incorrect', 'correct'};
miscDef= {'S251', 'S254';
          'start', 'end'};

mrk_stim= mrk_defineClasses(mrko, stimDef);
mrk_resp= mrk_defineClasses(mrko, respDef);
mrk_fb= mrk_defineClasses(mrko, feedbackDef);
mrk_err= mrk_defineClasses(mrko, errDef);
mrk_misc= mrk_defineClasses(mrko, miscDef);

% Correct for to many stimulations in start trials (due to adaptive mode?)
remEpos = [];
for i = 1:length(mrk_err.pos),
    nrToEnd = find(mrk_stim.toe(mrk_stim.pos < mrk_err.pos(i)));
    if length(nrToEnd) ~= i*90 + length(remEpos),
        remEpos = [remEpos nrToEnd(i*90 + length(remEpos)+1:end)];
    end
end
mrk_stim = mrk_chooseEvents(mrk_stim, 'not', remEpos);


if length(mrk_resp.pos) == 72,
    mrk_final_dec = mrk_selectEvents(mrk_resp, [1:2:72]);
    mrk_final_dec.className = {'classifier decision'};
    mrk_resp = mrk_selectEvents(mrk_resp, [2:2:72]);
elseif length(mrk_resp.pos) == 48,
    mrk_final_dec = [];
else
    error('number of counts is not correct');
end

alltargets = mrk_stim.toe(find(mrk_stim.toe > 10))-10;
targets = zeros(1,36);
tarCount = 1; targets(1) = alltargets(1); tarId = 2; maxTar = 17;
for i = 1:length(alltargets),
    if alltargets(i) ~= targets(tarId-1) || tarCount > maxTar,
        targets(tarId) = alltargets(i);
        tarId = tarId + 1;
        tarCount = 1;
    else
        tarCount = tarCount +1;
    end
end

% set targets correct/incorrect
if ~isempty(mrk_fb.toe) && length(mrk_fb.toe) == 432,
    tarId = 1; fb_count = 0;
    correct = zeros(1, length(mrk_fb.toe));
    for i= 1:length(mrk_fb.toe),
        if mrk_fb.toe(i)-50 == targets(tarId),
            correct(i) = 1;
        end
        fb_count = fb_count+1;
        if fb_count == 12,
            tarId = tarId+1;
            fb_count = 0;
        end
    end
    mrk_fb.correct_fb = correct;
    mrk_fb = mrk_addIndexedField(mrk_fb, 'correct_fb');
elseif ~isempty(mrk_fb.toe),
    warning('Not the correct number of tactor fb. Assuming training mode.');
end

% if length(mrk_resp.pos) > 0,
%     [mrk, istim, iresp]= ...
%         mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
%                           'missingresponse_policy', 'accept', ...
%                           'multiresponse_policy', 'first', ...
%                           'removevoidclasses', 0);
%         ivalid= find(~mrk.missingresponse);
%         hits = intersect(find(mrk.toe >10), ivalid);
%         mrk.ishit = zeros(1,length(mrk.toe));
%         mrk.ishit(hits) = 1;
%         mrk= mrk_addIndexedField(mrk, 'ishit');
%         mrk.misc= mrk_misc;
% else
%     mrk = mrk_stim;
% end

mrk = mrk_stim;
mrk.misc= mrk_misc;
if ~isempty(mrk_fb.toe)
    mrk.fb = mrk_fb;
end
mrk.err = mrk_err;
mrk.count = mrk_resp;
if exist('mrk_final_desc') && ~isempty(mrk_final_desc.toe),
    mrk.finaldec = mrk_final_dec;
end

