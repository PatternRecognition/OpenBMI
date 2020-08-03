function mrk = mrkodef_leitstand_season2(mrko, varargin)

% mrkodef_leistand_season2 - prepare markers for Leitstand Season 2
%
% Synopsis:
%   mrk = mrkodef_leitstand_season2(mrko, varargin)
%
% Arguments:
%   mrko - ???
%   varargin - ???
%
% Output:
%   mrk - ???
%
% Marker codings:
%
%
% Author(s) Bastian Venthur, 2010-02-19

% Benjamin:
% Ok, this is really bad, trying to make the best of of these shity
% markers.
% We start by removing markers, that can only be artifacts of too
% fast triggering.
black_list= cprintf('S%3d', [23 61 63 78 87 93 94 95 105 114 119 126 214 246 251 255]);
idel= [];
for ii= 1:length(black_list),
  idel= cat(1, idel, strmatch(black_list{ii}, mrko.desc));
end
mrko= mrk_chooseEvents(mrko, 'not',idel);

% We proceed by finding quick successions of two markers, where we know
% that the second one is an artifact and can be removed. 
pair_list= {'S 71','S 70';
            'S 72','S 70'};
dd= mrko.pos(2:end)-mrko.pos(1:end-1);
id= find(dd<10);
idel= [];
for kk= 1:length(id),
  for tt= 1:size(pair_list,1),
    if isequal(mrko.desc(id(kk)+[0:1]), pair_list(tt,:)),
      idel= [idel, id(kk)+1];
    end
  end
end
mrko= mrk_chooseEvents(mrko, 'not',idel);

% We proceed by finding quick successions of three markers, where we know
% that the middle one is an artifact and can be removed. 
triple_list= {'S 71','S111','S 41'};
dd= mrko.pos(3:end)-mrko.pos(1:end-2);
id= find(dd<20);
idel= [];
for kk= 1:length(id),
  for tt= 1:size(triple_list,1),
    if isequal(mrko.desc(id(kk)+[0:2]), triple_list(tt,:)) | ...
          isequal(mrko.desc(id(kk)+[0:2]), triple_list(tt,end:-1:1)),
      idel= [idel, id(kk)+1];
    end
  end
end
mrko= mrk_chooseEvents(mrko, 'not',idel);

black_list= cprintf('S%3d', [41]);
idel= [];
for ii= 1:length(black_list),
  idel= cat(1, idel, strmatch(black_list{ii}, mrko.desc));
end
mrko= mrk_chooseEvents(mrko, 'not',idel);


stimDef = {'S 20',      'S 21',     'S 22';
    'status',    'warning',  'alert'};

respDef = {'S 71',      'S 72',     'S 73',     'S 70';
    'r-correct', 'r-wrong',  'r-wrongblue', 'r-unknown'};

% The formular for the markers is: 100*base + 10*upper + lower
%   base: d = 100, p = 200
%   upper, lower: numbers of bars (0-2)
%d2StimDef = {'S101','S102', 'S110','S111','S112', 'S120','S121','S122', ...
%             'S201','S202', 'S210','S211','S212', 'S220','S221','S222';
%             'd01',  'd02',  'd10', 'd11', 'd12',  'd20', 'd21', 'd22', ...
%             'p01',  'p02',  'p10', 'p11', 'p12',  'p20', 'p21', 'p22'};

% Benjamin:
% We need a different d2StimDef, since some markers were modified due to
% too fast succession of triggers
d2StimDef = {'S100', 'S101', 'S102', 'S103', ...
             'S110', 'S111', 'S112', 'S113', ...
             'S120', 'S121', 'S122', 'S123', ...
             'S200', 'S201', 'S202', 'S203', ...
             'S210', 'S211', 'S212', 'S213', ...
             'S220', 'S221', 'S222', 'S223'};

d2RespDef = {'S 65',         'S 66';
    'd2_correct',   'd2_wrong'};

miscDef = {'S 99',  'S252',     'S253', 'S 50';
    'money', 'start',    'end',  'd2_start'};


opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'stimDef', stimDef, ...
    'respDef', respDef, ...
    'd2StimDef', d2StimDef, ...
    'd2RespDef', d2RespDef, ...
    'miscDef', miscDef);


mrk_misc = mrk_defineClasses(mrko, opt.miscDef);


% Hauptfenster
mrk_stim = mrk_defineClasses(mrko, opt.stimDef);
mrk_resp = mrk_defineClasses(mrko, opt.respDef);

% Fix to red/blue markers problem:
%
% Blue have the same markers as red, but should have their own ones.
% Fortunately, the real red markers are followed by a d2_start. So every red
% marker not followed by a d2_start is a blue one.
mrk_rot= mrk_selectClasses(mrk_stim, 'alert');
mrk_d2start= mrk_selectClasses(mrk_misc, 'd2_start');
[dmy, istim]= mrk_matchStimWithResp(mrk_rot, mrk_d2start);
[dmy, idxrot]= intersect(mrk_stim.pos, mrk_rot.pos(istim));
mrk_stim.y(3,:)= ismember(1:length(mrk_stim.pos), idxrot);
mrk_stim.y(1,:)= ~any(mrk_stim.y([2 3],:));
mrk_stim.toe(find(mrk_stim.y(1,:)))= 20;

%Benjamin:
% After each d2-test there is a S71 or S72 marker (maybe marking
%  the keypress of returning to the message screen?). We remove these
%  markers because they would otherwise result in 'reponse during empty
%  stack' warning in the subsequent part.
mrk_start_d2= mrk_selectClasses(mrk_misc, 'd2_start');
[dmy, dmy, iresp]= mrk_matchStimWithResp(mrk_start_d2, mrk_resp, ...
                                         'multiresponse_policy','first');
mrk_resp= mrk_chooseEvents(mrk_resp, 'not',iresp);

%Benjamin:
% Due due accumulation of messages, we cannot use the handy
% mrk_matchStimWithResp function, but rather have to do it by hand.
stack= [];
is= 1;
ir= 1;
idel_resp= [];
mrk= mrk_stim;
mrk.latency= NaN*zeros(1, length(mrk_stim.pos));
mrk.ishit= NaN*zeros(1, length(mrk_stim.pos));
mrk.missingresponse= ones(1, length(mrk_stim.pos));
mrk.messages_on_screen= zeros(1, length(mrk_stim.pos));
mrk_stim_pos= [mrk_stim.pos, inf];
mrk_resp_pos= [mrk_resp.pos, inf];
while is <= length(mrk_stim.pos) | ir <= length(mrk_resp.pos), 
  if mrk_stim_pos(is) < mrk_resp_pos(ir),
    % process next Stimulus marker
    mrk.messages_on_screen(is)= length(stack);
    stack= [stack, mrk_stim.toe(is)];
    is= is + 1;
  else
    % process next Response marker
    if isempty(stack),
      warning('response during empty stack encountered');
      idel_resp= [idel_resp, ir];
      ir= ir + 1;
      continue;
    end
    requires_response= find(ismember(stack, [21 22]));
    if isempty(requires_response),
      % in this case, the response is a false alarm
      ii= is - 1;
      mrk.ishit(ii)= 0;
      mrk.latency(ii)= (mrk_resp.pos(ir)-mrk_stim.pos(ii))/mrk.fs*1000;
      if mrk_resp.toe(ir)~=73,
        warning('positive response after blue message');
      end
    else
      ii= is - length(stack) + requires_response(1) - 1;
      mrk.latency(ii)= (mrk_resp.pos(ir)-mrk_stim.pos(ii))/mrk.fs*1000;
      if mrk_resp.toe(ir)~=70,
        mrk.ishit(ii)= mrk_resp.toe(ir)==71;
      end
      mrk.missingresponse(ii)= 0;
      stack(1:requires_response(1))= [];
    end
    ir= ir + 1;
  end
end
mrk= mrk_addIndexedField(mrk, {'latency','ishit','missingresponse', ...
                    'messages_on_screen'});


% D2-Test
mrk_d2Stim = mrk_defineClasses(mrko, opt.d2StimDef);
mrk_d2Resp = mrk_defineClasses(mrko, opt.d2RespDef);

% Benjamin:
% Remove artifact markers
id= find(diff(mrk_d2Stim.pos)<20);
mrk_d2Stim= mrk_chooseEvents(mrk_d2Stim, 'not',id+1);
id= find(diff(mrk_d2Resp.pos)<20);
mrk_d2Resp= mrk_chooseEvents(mrk_d2Resp, 'not',id+1);

% Benjamin:
% Now it is getting even more weird. It seems that the stimulus marker
% was set at the time point of the response.
[mrk.d2, istim, iresp] = mrk_matchStimWithResp(mrk_d2Stim, mrk_d2Resp, ...
    'multiresponse_policy', 'first', ...
    'min_latency', -20, ...
    'max_latency', 20, ...
    'removevoidclasses', 0);
% Benjamin:
% Do the best we can to obtain a latency for at least most stimuli:
latte= [NaN, (mrk.d2.pos(2:end)-mrk.d2.pos(1:end-1))/mrk.fs*1000];
invalid= find(latte>2000);  %% longer than 2s
latte(invalid)= NaN;
mrk.d2.latency= latte;


mrk.misc= mrk_misc;
