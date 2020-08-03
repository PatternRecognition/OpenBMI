function stim_visualOddballPTB(varargin)

% STIM_VISUALPODDBALL - Presents colored rectangle-stimuli in a visual oddball paradigm.
% You can also provide your own standard/target/deviant stimuli via the tx
% parameters in opt.
%
%Synopsis:
% stim_visualOddball(<OPT>)
%
%Arguments:
% OPT: struct or property/value list of optional arguments:
%
% 'prob'     : give probabilities for standard, target, and deviant as cell
%              array (default { .8 .2 0})
% 'txStandard' : double matrix in [0 1] specifying the standard
% 'txTarget'   : double matrix in [0 1] specifying the standard
% 'txDeviant'  : double matrix in [0 1] specifying the standard
% 'size'     : size [Width Height] of the rectangles
% 'colors'   : cell array giving the rectangle colors 
%              (default {[0 0 255] [255 0 0] [0 255 0]})
% 'time'     : presentation time of each stim (default 0.2 s)
% 'ISI'      : time between OFFset of one stim to ONset of next stim
%              (default 1 s)
% 'timeJitter' : if x, on each trial a random time offset in the range[0 x] 
%                will be added to the ISI variable (default 0)
% 'nStim'    : total number of stimulus presentations (default 10)
% 'filename' : Filename of EEG file
% 'test'     : if 1 EEG data is recorded (default 0)
% 'fixdot'   : radius of fixation dot (default 0)
% 'fixdotColor' : color of fixation dot as [r g b] (default [0 0 255])
% 'bg'       : color of screen background (default black)
% 'countdown': the number of secs for the countdown # TO DO
% 'screenid' : if you want to open your window not on the standard screen
%              (0), provide the screenid here. You can use the PTB-3
%              command myscreens = Screen('Screens') to enumerate all
%              possible screens.
% 'fullscreen' : specify if you want to run in fullscreen mode (default 0)
%
% Triggers sent: 252 (startup), 253 (end presentation), 10.... (standards)
% 30 .... (deviants), 50 .... (targets)
%
% Note: Psychophysics toolbox (PTB-3) is needed to run this script! You can
% get the toolbox for free at http://psychtoolbox.org.
%
% Example
% opt=struct('bv_host',[],'test',1);
% stim_visualOddball(opt);
% 
% Matthias Treder 24-08-2009

global VP_SCREEN VP_CODE

TRIG_START = 252; TRIG_END = 253;
TRIG_STAN = 10; TRIG_DEV = 30; TRIG_TAR = 50;

%% Process OPT and set default
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'bv_host', 'localhost', ...
                  'filename', 'arte', ...
                  'position', VP_SCREEN, ...
                  'prob',{.4 .4 .2},...
                  'txStandard',[],...
                  'txTarget',[],...
                  'txDeviant',[],...
                  'time', .1,...
                  'ISI', 1,...
                  'timeJitter', .3,...
                  'nStim', 10,...
                  'size', [200 60], ...
                  'colors', {[0 0 255] [255 0 0] [0 255 0]},...
                  'fixdot',0,...
                  'fixdotColor',[0 0 255],...
                  'bg',[0 0 0 ],...
                  'countdown',3,...
                  'screenid',0,...
                  'fullscreen',0,...
                  'test', 0);

setupScreenPTB;

%% Setup BVR
if ~isempty(opt.bv_host),
  bvr_checkparport;
end

if ~opt.test
  if ~isempty(opt.filename)
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty')
  end
end


%% Make standard, target, and deviant & turn into openGL texture
wcol  = WhiteIndex(win); % pixel value for white
if ~isempty(opt.txStandard)
    txStandard = Screen('MakeTexture', win, opt.txStandard*wcol);
else
    txStandard = Screen('OpenOffscreenWindow',-1, opt.bg, [0 0 opt.size(1) opt.size(2)]);
    Screen('FillRect', txStandard, opt.colors{1},[0 0 opt.size(1) opt.size(2)]);
end
if ~isempty(opt.txTarget)
    txTarget = Screen('MakeTexture', win, opt.txTarget*wcol);
else
    txTarget= Screen('OpenOffscreenWindow',-1, opt.bg, [0 0 opt.size(1) opt.size(2)]);
    Screen('FillRect', txTarget, opt.colors{2},[0 0 opt.size(1) opt.size(2)]);
end
if ~isempty(opt.txDeviant)
    txDeviant = Screen('MakeTexture', win, opt.txDeviant*wcol);
else
    txDeviant = Screen('OpenOffscreenWindow',-1, opt.bg, [0 0 opt.size(1) opt.size(2)]);
    Screen('FillRect', txDeviant, opt.colors{3},[0 0 opt.size(1) opt.size(2)]);
end
%% Cycle
ppTrigger(TRIG_START);
% Present countdown
if ~isempty(opt.countdown)
    stim_countdownPTB
end

% Synchronize to retrace at start of trial/animation loop:
vbl = Screen('Flip', win);
% Present fixation
Screen('FillRect',win, opt.bg);       % Background
if opt.fixdot > 0, 
    Screen('FillOval', win, opt.fixdotColor,CenterRect([0 0 opt.fixdot opt.fixdot], winRect));       % Fixation dot
end
vbl = Screen('Flip',win,vbl + opt.ISI + rand()*opt.timeJitter - slack);

for k=1:opt.nStim
    %% Randomly choose next stimulus
    r = rand();
    if r <= opt.prob{1}
        nextStim = 1;
    elseif r <= opt.prob{1}+opt.prob{2} 
        nextStim = 2;
    else nextStim = 3;
    end
    Screen('FillRect',win, opt.bg);       % Background
    %% Draw stimulus 
    if nextStim == 1, Screen('DrawTexture',win,txStandard);        
    elseif nextStim == 2, , Screen('DrawTexture',win,txTarget);
    else Screen('DrawTexture',win,txDeviant);
    end
    %% Wait ISI before drawing
    vbl = Screen('Flip',win,vbl + opt.ISI + rand()*opt.timeJitter - slack);
    %% Send trigger
    if nextStim==1 
        ppTrigger(TRIG_STAN);
    elseif nextStim==2
        ppTrigger(TRIG_TAR);
    else
        ppTrigger(TRIG_DEV);
    end
    %% Wait for opt.time and draw blank screen
    Screen('FillRect',win, opt.bg);       % Background
    vbl = Screen('Flip',win,vbl + opt.time + rand()*opt.timeJitter - slack);

end
pause(opt.time)


%% Shut down
ppTrigger(TRIG_END);
Screen('CloseAll')