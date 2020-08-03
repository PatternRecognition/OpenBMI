function todo = stim_artifactMeasurement(string, sounds, varargin)
%todo = artefact(string, sounds, fixFcn)
%
% IN: string -  defines the sequence of acoustic stimuli
%               RULES: S -> 'F'#  (to play file [#] async)
%                      S -> 'f'# (to play file [#] sync)
%                      S -> P# (pause, in msec)
%                      S -> *(S) (more signals)
%                      S -> R[#](S) (to repeat a signal)
%     sounds -  cell array of sounds
%     fixFcn -  function to be called in the beginning, default fixCross
%
% OUT: todo - parsed string  (just for debugging purpose)
%
% EXAMPLE:
%  see setup_artifacts, setup_koffer_artifacts

global VP_CODE VP_SCREEN TODAY_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'bv_host', 'localhost', ...
                  'filename', 'arte', ...
                  'position', VP_SCREEN, ...
                  'multicross', 1, ...
                  'opt_animation', [], ...
                  'handle_background', 0, ...
                  'test', 0, ...
                  'useSignalServer', 0, ...
                  'delete_obj', []);
              
              
if opt.useSignalServer
    disp('SignalServer should be running already. Please check!');
else
    if ~isempty(opt.bv_host),
%        bvr_checkparport;
    end
end

delete(opt.delete_obj(find(ishandle(opt.delete_obj))));
if ~ishandle(opt.handle_background),
  opt.handle_background= stimutil_initFigure(opt);
end
if ~isfield(opt, 'handle_cross') | isempty(opt.handle_cross),
  if opt.multicross,
    opt.handle_cross= stimutil_fixationCrosses(opt);
  else
    opt.handle_cross= stimutil_fixationCross(opt);
  end
  set(opt.handle_cross, 'Visible','on');
end

if ~opt.test,
    if ~isempty(opt.filename),
        if opt.useSignalServer
            signalServer_startrecoding([opt.filename VP_CODE]); % ToDO: veraendern, wenn storeData an bvr_startrecording angepasst ist
        else
        bvr_startrecording([opt.filename VP_CODE]);
        end
    else
        warning('!*NOT* recording: opt.filename is empty');
  end
end

str = {};
i = 1;
while i<=length(string)
  if isempty(str2num(string(i)));
    if string(i)~=' '
      str = {str{:},string(i)};
    end 
    i = i+1;
  else
    i_st = i;
    i = i+1;
    while i<=length(string) && ~isempty(str2num(string(i)));
      i = i + 1;
    end
    str = {str{:},str2num(string(i_st:i-1))};
  end
end



todo = parse(str,0);
ppTrigger(252);
pause(0.25);
if sum(todo(1,:)==4)>0,
  H_anim= stimutil_animation('init', opt.opt_animation);
end

for i = 1:size(todo,2);
  switch(todo(1,i))
   case 1
    ppTrigger(todo(2,i));
    wavplay(sounds(todo(2,i)).sound,sounds(todo(2,i)).fs,'async');
   case 2
    ppTrigger(todo(2,i));
    wavplay(sounds(todo(2,i)).sound,sounds(todo(2,i)).fs,'sync');
   case 3
    pause(todo(2,i)/1000);
   case 4,
    stimutil_animation(todo(2,i)/1000);
    set(H_anim.patch, 'Visible','off');
   case 5,
    stimutil_countdown(todo(2,i)/1000, 'countdown_msg','%d', ...
                       'countdown_fontsize', 0.2);
  end 
end

ppTrigger(253);
pause(1);
if ~opt.test & ~isempty(opt.filename),
    if ~opt.useSignalServer
        bvr_sendcommand('stoprecording');
    else
        ppTrigger(254)
    end
end

if nargout==0,
  clear todo;
end



function [a,rest] = parse(str,rek);

if isempty(str)
  if rek>0
    error('parsing error');
  end
  a = [];
  rest = '';
  return;
end

if isnumeric(str{1})
  error('parsing error');
end

if str{1}=='F';
  [a,rest] = parse(str(3:end),rek);
  a = cat(2,[1;str{2}],a);
elseif str{1}=='f';
  [a,rest] = parse(str(3:end),rek);
  a = cat(2,[2;str{2}],a);
elseif str{1}=='P'
  [a,rest] = parse(str(3:end),rek);
  a = cat(2,[3;str{2}],a);
elseif str{1}=='A'
  [a,rest] = parse(str(3:end),rek);
  a = cat(2,[4;str{2}],a);
elseif str{1}=='C'
  [a,rest] = parse(str(3:end),rek);
  a = cat(2,[5;str{2}],a);
elseif str{1}==')' 
  if rek==0
    error('parsing error');
  end
  a = [];
  rest = str(2:end);
  
elseif str{1}=='R'
  if str{2}~='[' | ~isnumeric(str{3}) | str{4}~=']' | str{5}~='('
    error('parsing error');
  end
  [b,rest] = parse(str(6:end),rek+1);
  b = repmat(b,[1,str{3}]);
  a = cat(2,b,parse(rest,rek));
else
  error('parsing error');
end
