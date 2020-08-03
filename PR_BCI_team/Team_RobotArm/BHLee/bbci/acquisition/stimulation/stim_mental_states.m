function stim= stim_mental_states(state, varargin)
%stim= mental_states(state, <opt>)

%     opt
%             .minEventsLeft - break only if at least this many events left

%global IO_ADDR 
global SOUND_DIR

if ~isstruct(state),
  error('first argument must be a struct');
end
if ~isfield(state, 'param'),
  [state.param]= deal({});
end
if ~isfield(state, 'presentation_fcn'),
  [state.presentation_fcn]= deal({});
else
  tmp= apply_cellwise({state.stimulus}, 'length');
  if any([tmp{:}])>1,
    opt.presentation_fcn= state.stimulus;
  end
end
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'bkgColor',0.3*[1 1 1], ...
  'textColor',0*[1 1 1], 'textPos',0.55, 'textSize',0.1, ...
  'crossColor',0.7*[1 1 1], 'crossLineWidth',4, ...
  'stimPos',0.55, 'stimSize',0.25, 'stimColor',[0 0 0], ...
  'maxExtent',1, ...
  'trialTiming',[3 1 0.5], ...
  'breakFreq',inf, ...
  'minEventsLeft',7, ...
  'relaxTimeBetween',[], ...
  'breakFactor',1, ...
  'msgRelax','entspannen', ...
  'msgBreak','kurze Pause', ...
  'msgContinue','Taste drücken um fortzufahren',...
  'speech_dir', [SOUND_DIR 'english\'], ...
  'auditory',0);
%opt.stimColor= [0.6 0.1 0.1];  %% for Klaus


if length(opt.breakFreq)==1,
    opt.breakFreq= [opt.breakFreq 10];
end
if length(opt.relaxTimeBetween)==1,
  opt.relaxTimeBetween= [opt.relaxTimeBetween, opt.trialTiming(2:3)];
end
if length(opt.trialTiming)==3
    opt.trialTiming(4)=0;
end 
%opt.msgBreak= 'short break';
%opt.msgRelax= 'relax';
if opt.auditory
    sounds = struct('file',[],'fs',[]);
    for i = 1:length(state)
        switch(state(i).stimulus)
            case 'L'
                [sounds(i).file,sounds(i).fs] = wavread([SPEECH_DIR 'speech_left']);
            case 'R'
                [sounds(i).file,sounds(i).fs] = wavread([SPEECH_DIR 'speech_right']);
            case 'T'
                [sounds(i).file,sounds(i).fs] = wavread([SPEECH_DIR 'speech_tonge']);
            case 'F'
                [sounds(i).file,sounds(i).fs] = wavread([SPEECH_DIR 'speech_foot']);
            case 'B'
                [sounds(i).file,sounds(i).fs] = wavread([SPEECH_DIR 'speech_both']);
            case 'X'
                [sounds(i).file,sounds(i).fs] = wavread([SPEECH_DIR 'speech_relax']);
        end
    end 
    i= length(state);
   [sounds(i+1).file,sounds(i+1).fs] = wavread([SPEECH_DIR 'speech_stop']);
   [sounds(i+2).file,sounds(i+2).fs] = wavread([SPEECH_DIR 'speech_close_eyes']);
   [sounds(i+3).file,sounds(i+3).fs] = wavread([SPEECH_DIR 'speech_open_eyes']);
   for ii= 1:3,
       [sound_counting(ii).wav, sound_counting(ii).fs]= ...
       wavread([SPEECH_DIR 'speech_' int2str(ii) '.wav']);
   end
end 

nClasses= length(state);
nEvents= sum([state(:).nEvents]);

events= [];
for cc= 1:nClasses,
    events= cat(1, events, cc*ones(state(cc).nEvents,1));
end
events= events(randperm(nEvents));
nBreaks= floor((nEvents-opt.minEventsLeft)/opt.breakFreq(1));
fprintf('To stop during ''%s'' press space-bar in activated figure.\n', opt.msgBreak);
tt= opt.trialTiming(1:3);
if ~isempty(opt.relaxTimeBetween),
  tt= tt+opt.relaxTimeBetween;
end
fprintf('estimated experiment time: %.2f min [+ %.2f min pause]\n', ...
        nEvents*tt*[1;1;0.5]/60, ...
        (2+58*opt.breakFactor)/60 + nBreaks*(opt.breakFreq(2)+5)/60);

oldPos= figureMaximize;
set(gcf, 'color',opt.bkgColor);
set(gcf,'KeyPressFcn','global pausi;pausi = [pausi,get(gcf,''CurrentCharacter'')];');
set(gcf,'DoubleBuffer','on');

if opt.breakFactor>=0,
  ppTrigger(251);
  if opt.auditory
      wavplay(sounds(length(state)+2).file,sounds(length(state)+2).fs,'async');
  end   
  h_text= showText(opt.msgRelax, opt.textSize, opt.textPos, opt.textColor);
  set(h_text, 'color', opt.textColor); 
  pause(1+39*opt.breakFactor);
  set(h_text, 'string',' ');
  h_cross= line([.48 .52; .5 .5]', 0.05+[.5 .5; .475 .525]', ...
                'color',opt.crossColor, 'lineWidth',opt.crossLineWidth); 
  axis([0 1 0 1]);
  pause(1);
  
  if opt.auditory,
    wavplay(sound_counting(3).wav, sound_counting(3).fs, 'async');
  end
  set(h_text, 'string','start in 3 s'); 
  pause(1);
  if opt.auditory,
    wavplay(sound_counting(2).wav, sound_counting(2).fs, 'async');
  end
  set(h_text, 'string','start in 2 s'); 
  pause(1);
  if opt.auditory,
    wavplay(sound_counting(1).wav, sound_counting(1).fs, 'async');
  end 
  set(h_text, 'string','start in 1 s'); 
  pause(1);
  set(h_text, 'string',' ');
  
  ppTrigger(252);
  pause(1);
end

h_stim= showText(' ', opt.stimSize, opt.stimPos, opt.textColor);
set(h_stim, 'color', opt.stimColor); 

% move h_stim to the background (behind the cross)
hc= get(gca, 'children');
set(gca, 'children',hc([2:end 1]));

h_imageax= [];
stim= cell(1, nEvents);
for ei= 1:nEvents,
  cc= events(ei);
  if isempty(state(cc).presentation_fcn),
    stim{ei}= state(cc).stimulus;
  else
    stim{ei}= feval(state(cc).stimulus, state(cc).param{:});
  end
  if ischar(stim{ei}),
      if opt.auditory
          sound_ind = strmatch(stim{ei},{state.stimulus});
          wavplay(sounds(sound_ind).file,sounds(sound_ind).fs,'async'); 
      end   
    set(h_stim, 'string', stim{ei}, 'fontSize',opt.stimSize);
    ext= get(h_stim, 'extent');
    while ext(3)>opt.maxExtent,
      set(h_stim, 'fontSize',0.95*get(h_stim,'fontSize'));
      ext= get(h_stim, 'extent');
    end
  else
    set(h_stim, 'string',' ');
    if isempty(h_imageax),
      h_imageax= axes('position',[0.5-0.5*opt.imageExtent(1), ...
        opt.stimPos-0.5*opt.imageExtent(2), ...
        opt.imageExtent(1), opt.imageExtent(2)]);
      axis('equal');
      colormap([opt.bkgColor; opt.stimColor]);
    end
    axes(h_imageax);
    h_im= imagesc(stim{ei});
    set(h_imageax, 'visible','off');
    axis('equal');
    moveObjectBack(h_imageax);
  end
  ppTrigger(events(ei));
  pause(opt.trialTiming(1)+opt.trialTiming(4)*rand);
  if ischar(stim{ei}),
    set(h_stim, 'string',' ');
    if opt.auditory
        wavplay(sounds(length(state)+1).file,sounds(length(state)+1).fs,'async');
    end 
  else
    delete(h_im);
  end
  ppTrigger(100);
  pause(opt.trialTiming(2)+opt.trialTiming(3)*rand);
  if ~isempty(opt.relaxTimeBetween),
    set(h_stim, 'string', 'x', 'fontSize',opt.stimSize);
    ppTrigger(nClasses+1);
    pause(opt.relaxTimeBetween(1));
    set(h_stim, 'string', ' ');
    ppTrigger(100);
    pause(opt.relaxTimeBetween(2)+opt.relaxTimeBetween(3)*rand);
  end
   
  if mod(ei, opt.breakFreq(1))==0 & ei<nEvents-opt.minEventsLeft & ...
    opt.breakFactor>=0,
    if opt.auditory
      give_him_a_break(opt.breakFreq(2), h_text, sounds((end-1):end), opt);
    else  
       give_him_a_break(opt.breakFreq(2), h_text, [], opt);
    end   
  end
end

delete([h_stim; h_cross]);
set(h_text, 'string',opt.msgRelax);
if opt.auditory
  wavplay(sounds(length(state)+2).file,sounds(length(state)+3).fs,'async');
end   
ppTrigger(253);
pause(1+9*opt.breakFactor);

ppTrigger(254);
clf;
figureRestore(oldPos);




function give_him_a_break(breakLength, h_text, sounds, opt)

global pausi
ppTrigger(249);
set(h_text, 'string',opt.msgBreak);
pausi = '';
  if opt.auditory
      wavplay(sounds(2).file,sounds(2).fs,'async');
  end   
pause(breakLength);
if strfind(pausi,' ');
    set(h_text,'string',opt.msgContinue);
    pause; 
end
set(h_text, 'string',' ');
pause(1);
  if opt.auditory
      wavplay(sounds(1).file,sounds(1).fs,'async');
  end   
anounce_start(h_text);
ppTrigger(250);
pause(1);
