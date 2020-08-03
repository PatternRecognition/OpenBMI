function [correct,response,lt,br]= stim_d2test(N, varargin)
%STIM_D2TEST

global VP_CODE VP_SCREEN

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filename', 'd2test', ...
                  'position', VP_SCREEN, ...
                  'test', 0, ...
                  'countdown', 7, ...
                  'perc_dev', 0.5, ...
                  'response_markers', {'R 16', 'R  8'}, ...
                  'bv_host', 'localhost', ...
                  'duration_cue', 250, ...
                  'countdown', 7, ...
                  'timeout', inf,...
                  'response_delay', [400 1100], ...
                  'duration_response', 750, ...
                  'duration_blank', 250, ...
                  'duration_intro', 7000, ...
                  'msg_intro','Entspannen', ...
                  'msg_missed','verpasst');

plett= 'd';
nlett= 'b';

if length(opt.response_delay)==1,
  opt.response_delay= [1 1]*opt.response_delay;
end

if ~isempty(opt.bv_host),
  bvr_checkparport;
end

opt.handle_background= stimutil_initFigure(opt);
h_msg= stimutil_initMsg(opt);
set(h_msg, 'String',opt.msg_intro, 'Visible','on');
h_missed= stimutil_initMsg(opt);
set(h_missed, 'Visible','off', 'String',opt.msg_missed, 'Color','r');
h_reactometer= stimutil_reactometer('init', opt);
axes(opt.handle_background);
drawnow;
waitForSync;

if opt.test,
  fprintf('Warning: test option set true: EEG is not recorded!\n');
  pause(1);
else
  if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty');
  end
  ppTrigger(251);
  waitForSync(opt.duration_intro);
end

if ~isfield(opt, 'handle_cross') | isempty(opt.handle_cross),
  opt.handle_cross= stimutil_fixationCross(opt);
else
  set(opt.handle_cross, 'Visible','on');
end
set(h_msg, 'Visible','off');
set(h_reactometer(2:end), 'Visible','on');

if ~opt.test,
  pause(1);
  stimutil_countdown(opt.countdown, opt);
  ppTrigger(252);
end
pause(1);

[lt,br]= generate_d2_cue_sequence(N, plett, nlett, opt.perc_dev);
response= zeros(N, 1);
correct= NaN*zeros(N, 1);
h_d2= [];
state= acquire_bv(1000, opt.bv_host);
waitForSync;
for ei= 1:N,
  barCode= br(ei,:) .* 2.^(1:4);
  [dmy]= acquire_bv(state);  %% clear the queue
  ppTrigger(128+ismember(lt(ei),plett)+sum(barCode));
  h_d2= stimutil_showD2cue(lt(ei), br(ei,:), 'handle_d2',h_d2);
  drawnow;
  t0= clock;
  resp= [];
  if opt.duration_cue < opt.timeout,
    waitForSync(opt.duration_cue);
    set(h_d2, 'Visible','off');
    ppTrigger(100);
    drawnow;
  else
    while isempty(resp) & 1000*etime(clock,t0)<opt.timeout,
      [dmy,bn,mp,mt,md]= acquire_bv(state);
      for mm= 1:length(mt),
        resp= strmatch(mt{mm}, opt.response_markers);
        if ~isempty(resp),
          continue;
        end
      end
      pause(0.001);  %% this is to allow breaks
    end
    response(ei)= etime(clock,t0);
    delete(h_d2);
    drawnow;
  end
  waitForSync(opt.response_delay(1)+rand*diff(opt.response_delay));

  if opt.duration_cue < opt.timeout,
    [dmy,bn,mp,mt,md]= acquire_bv(state);
    iStim= strmatch('S', mt);
    iStim= iStim(1);
    if isempty(iStim),
      warning('could not find stimulus marker');
    end
    
    locked = 0;
    for mm= 1:length(mt),
      if locked == 0
        resp= strmatch(mt{mm}, opt.response_markers);
        
        if ~isempty(resp),
          iResp= mm;
          locked = 1;
          continue;
        end
      end
    end
    if isempty(resp),
      response(ei)= NaN;
    else
      response(ei)= (mp(iResp)-mp(iStim))/1000;
    end
  end

  if ~isempty(resp),
    pos= lt(ei)==plett & sum(br(ei,:))==2;
    correct(ei)= (pos & resp==1) | (~pos & resp==2);
    stimutil_reactometer(1000*response(ei)*sign(correct(ei)-0.5));
  else
    set(h_missed, 'Visible','on');
    drawnow;
  end
  ppTrigger(101);
  fprintf('%4d: %4.0fms  ->  %d:%d  (%d missed),  err= %.0f%%\n', ...
    ei, 1000*response(ei), sum(correct(1:ei)==1), sum(correct(1:ei)==0), ...
    sum(isnan(correct(1:ei))), 100*mean(correct(1:ei)==0));
  waitForSync(opt.duration_response);
  set([h_reactometer(1); h_missed], 'Visible','off');
  drawnow;
  ppTrigger(102);
  waitForSync(opt.duration_blank);
end
acquire_bv('close');

ppTrigger(253);
pause(1);

msg= sprintf('%d hits  :  %d errors  (%d missed)', ...
  sum(correct==1), sum(correct==0), sum(isnan(correct)));
set(h_msg, 'String',msg, ...
           'Visible','on', ...
           'FontSize', 0.5*get(h_msg,'FontSize'));

iGood= find(correct==1);
iBad= find(correct==0);
fprintf('%s\n', msg);
fprintf('|  average response time:  ');
if ~isempty(iGood),
   fprintf('%.2f s on hits  |', mean(response(iGood)));
end
if ~isempty(iBad),
   fprintf('%.2f s on errors  |', mean(response(iBad)));
end
fprintf('\n');

ppTrigger(254);
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(5);
delete(h_msg);

return





function [letters,bars] = ...
    generate_d2_cue_sequence(m,plett,nlett,pfrac,pbars,plinnfrac)
%GENERATE_D2_CUE_SEQUENCE produces randomly instances of the d2test
%
%   a = randd2(m,plett,nlett,pfrac,pbars,plinnfrac)
%
%     input:   m         - the number of instances
%              plett     - row vector of good letters
%              nlett     - row vector of bad letters (disjunct to plett)
%              pfrac     - fraction of good instances, default .5
%              pbars     - list of good amounts of bars, default 2
%              plinnprob - propability of good letters in neg instances,
%                          default ratio of #plett and #total letters
%     output:  letters   - (m,1)-matrix of letters
%              bars      - (m,4)-matrix of 0 and 1
%
%   STH * 14DEC2000

if ~exist('plinnfrac', 'var'),
  plinnfrac= length(plett)/length([plett nlett]);
end
if ~exist('pfrac', 'var') | isempty(pfrac), pfrac=1/2; end
if ~exist('pbars') | isempty(pbars), pbars=2; end

nbars = 4;  % how many bars are there?
letters = zeros(m,1);
bars    = zeros(m,nbars);
% we produce randomly instances until we have enough,
% with weird parameters this might take VERY long

% first we produce positive ones
n = 1;
while n <= pfrac*m
   letters(n,1) = plett(ceil(rand*length(plett)));
   ok = 0;
   while ~ok
      bars(n,:) = floor(2*rand([1 nbars]));    % maximal four bars
      ok= ismember(sum(bars(n,:)),pbars);
   end
   n = n + 1;
end
% and the rest negative ones
while n <= m
   if rand < plinnfrac
     letters(n,1) = plett(ceil(rand*length(plett)));
   else
     letters(n,1) = nlett(ceil(rand*length(nlett)));
   end
   ok = 0;
   while ~ok
      bars(n,:) = floor(2*rand([1 nbars]));    % maximal four bars
      ok= ismember(letters(n,1),nlett) | ~ismember(sum(bars(n,:)),pbars);
   end
   n = n + 1;
end
% finally we have to mix up all instances
p = randperm(m);
letters = char(letters(p,:));
bars = bars(p,:);
