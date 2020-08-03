%% Falls Probleme, VLC Einstellungen ueberpruefen
% Menu -> Interface -> Einstellungen
% Interface -> Minimalansicht-Modus -> EIN
% Einstellungen zeigen (u.links) -> Alle
% Video ->  Videotitel fuer x ms zeigen -> 0
% Video -> Automatische Videoskalierung -> AUS
% Video -> Videoausrichtung -> zentriert
%

fprintf('Plug in light diode. Press <RETURN> when finished.\n');
pause

dbstop if error
VP_SCREEN = [-1920 0 1920 1200];

%% Trigger and timing
RUN_START = 254;
RUN_END = 255;
TRIAL_START = 250;
TRIAL_END = 251;
PAUSE_START = 248;
PAUSE_END = 249;

tAfter = 2;   % time in s after each video

%% Video settings 
vlcdir = 'C:\Program Files\VideoLAN\VLC';
videodir = 'D:\data\hhi_videos\';  % directory that is scanned for vids
% VLC options
videoopt = [' --play-and-exit --no-loop --no-video-deco --no-autoscale --scale 2 --no-media-library -I dummy --no-embedded-video --width=2560 --height=1600'];
% Videohöhe: 24cm --> Sichtabstand: 4*h= 96cm
vids = dir(fullfile(videodir,'*.avi'));
vids = struct2cell(vids);
vidnames = vids(1,:);
for n = 1:length(vidnames)
  vidnames{n} = get_prefix(vidnames{n},'_');
end
vidtypes = unique(vidnames);
vids = vids(1,:);
vids = vids(randperm(length(vids)));


%% Experiment
fprintf('Press key to start the actual experiment... \n') %TODO; Jets_832x480_55_HQ_trig_276_LQ3.avi
pause
bvr_startrecording(['video' VP_CODE]);
pause(5);
ppTrigger(RUN_START);

v=1;
while v<=length(vids)
vidname = [videodir vids{v}];
cmd= ['cmd /C "C: & cd ' vlcdir ' & vlc ' videoopt ' ' vidname];
vidID = strmatch(get_prefix(vids{v}),vidtypes)
ppTrigger(vidID);
pause(0.1);
vidname2ppcode(vids{v})
ppTrigger(vidname2ppcode(vids{v}));
system(cmd);           % show video
system('exit');
ppTrigger(TRIAL_END);
pause(tAfter);
if mod(v,43)==0   % long pause every ~12.5 minutes
  ppTrigger(RUN_END);
  bvr_sendcommand('stoprecording');
  disp([int2str(length(vids)-v) ' videos to go']);
  keyboard
  if length(vids)-v==0
    break
  end
  bvr_startrecording(['video' VP_CODE]);
  ppTrigger(RUN_START); pause(1)
end
v=v+1;
end


%%
fprintf('Experiment finished.\n');




