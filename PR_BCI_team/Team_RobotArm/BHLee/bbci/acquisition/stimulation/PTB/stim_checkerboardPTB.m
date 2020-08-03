function stim_checkerboardPTB(varargin)

% STIM_CHECKERBOARDPTB  - Presents a black-white, contrast-reversing 
% checkerboard pattern.
%
%Synopsis:
% stim_checkerboardPTB(<OPT>)
%
%Arguments:
% OPT: struct or property/value list of optional arguments:
% 'checkSize': size (length of the vertical/horizontal) of each checker in
%              pixels
% 'nChecks'  : number of checks along the y and x dimension (must be even!)
% 'time'     : time from one reversal to next (default 1.2 s)
% 'timeJitter' : if x, on each trial a random time offset in the range[0 x] 
%                will be added to the time variable (default 0)
% 'nStim'    : total number of stimulus presentations (default 10)
% 'filename' : Filename of EEG file
% 'test'     : if 1 EEG data is recorded (default 0)
% 'fixdot'   : radius of fixation dot (default 0)
% 'fixdotColor' : color of fixation dot as [r g b] (default [0 0 255])
% 'bg'       : color of screen background (default gray)
% 'countdown': the number of secs for the countdown
% 'screenid' : if you want to open your window not on the standard screen
%              (0), provide the screenid here. You can use the PTB-3
%              command myscreens = Screen('Screens') to enumerate all
%              possible screens.
% 'fullscreen' : specify if you want to run in fullscreen mode (default 0)
% 'texture'  : you can provide your own two textures here if you do not
%              want to use checkerboard. texture should be a double matrix
%              with color intensities normalized to [0 1]. Provide the
%              textures as a cell array so that .texture{1} is the first
%              and .texture{2} is the second texture.
%
% Triggers sent: 252 (startup), 253 (end presentation), 20 (pattern
% reversal 1), 21 (pattern reversal 2)
%
% Note: Psychophysics toolbox (PTB-3) is needed to run this script! You can
% get the toolbox for free at http://psychtoolbox.org.
%
% Example
% opt=struct('bv_host',[],'test',1);
% stim_checkerboardPTB(opt);
% 
% Matthias Treder 24-08-2009

global VP_SCREEN VP_CODE

TRIG_START = 252; TRIG_END = 253;
TRIG_CHECK = [20 21];

%% Process OPT and set default
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'bv_host', 'localhost', ...
                  'filename', 'arte', ...
                  'position', VP_SCREEN, ...
                  'time', 1.2,...
                  'timeJitter', 0,...
                  'nStim', 10,...
                  'checkSize', 60, ...
                  'nChecks', [10 10], ...
                  'fixdot',0,...
                  'fixdotColor',[0 0 255],...
                  'bg',[128 128 128],...
                  'countdown',3,...
                  'screenid',0,...
                  'fullscreen',0,...
                  'texture',[],...
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

%% Make checkerboard & turn into openGL texture
wcol  = WhiteIndex(win); % pixel value for white
if isempty(opt.texture)
    % Generate own checkerboard texture
    c  = double(checkerboard(opt.checkSize,opt.nChecks(1)/2,opt.nChecks(2)/2)>.5);
    tx(1) = Screen('MakeTexture',win,c*wcol);
    tx(2) = Screen('MakeTexture',win,double(1-c)*wcol);
else
    % Generate textures from matrices provided in opt.texture
    tx(1) = Screen('MakeTexture',win,opt.texture{1}*wcol);
    tx(2) = Screen('MakeTexture',win,opt.texture{2}*wcol);
end

%% Cycle
ppTrigger(TRIG_START);

% Present countdown
if ~isempty(opt.countdown)
    stim_countdownPTB
end
% Synchronize to retrace at start of trial/animation loop:
vbl = Screen('Flip', win);
% Present fixation dot for 1 s
Screen('FillRect',win, opt.bg);       % Background
Screen('FillOval', win, opt.fixdotColor,CenterRect([0 0 20 20], winRect));       % Fixation dot
vbl = Screen('Flip',win);

for k=1:opt.nStim
    Screen('FillRect',win, opt.bg);       % Background
    Screen('DrawTexture',win,tx(mod(k,2)+1));        
    Screen('FillOval', win, opt.fixdotColor,CenterRect([0 0 20 20], winRect));       % Fixation dot
    vbl = Screen('Flip',win,vbl + opt.time + rand()*opt.timeJitter - slack);
    ppTrigger(TRIG_CHECK(mod(k,2)+1));
end
pause(opt.time)

%% Shut down
ppTrigger(TRIG_END);
Screen('CloseAll')
