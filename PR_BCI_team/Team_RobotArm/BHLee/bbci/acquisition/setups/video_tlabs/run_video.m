%% Falls Probleme, VLC Einstellungen ueberpruefen
% Menu -> Interface -> Einstellungen
% Interface -> Minimalansicht-Modus -> EIN
% Einstellungen zeigen (u.links) -> Alle
% Video ->  Videotitel fuer x ms zeigen -> 0
% Video -> Automatische Videoskalierung -> AUS
% Video -> Videoausrichtung -> zentriert
%

%% Settings 
fprintf('Plug in light diode. Press <RETURN> when finished.\n');
pause
dbstop if error

%% Example Trials
fprintf('Press <RETURN> to start the example trials.\n'); pause;

ex_vids = {'tiling01.avi', 'tiling05.avi', 'tiling10.avi'};
nTrialsPerCondition = 3;
nConditions = length(ex_vids);
stim_seq = mod(0:nTrialsPerCondition*nConditions-1,nConditions)+1;
stim_seq = stim_seq(randperm(length(stim_seq)));

% Make connection to bv recorder for obtaining trigger data
state= acquire_bv(1000, 'localhost');
state.reconnect= 1;

for n = 1:length(stim_seq)
    vid = ex_vids{stim_seq(n)};
    responded = 0;
    % start video
    vidname = [videodir vid];
    cmd = ['cd ' VLC_DIR ' & vlc ' VLC_OPTS ' ' vidname];
    dos(cmd);
    while  ~responded
        s = stimutil_waitForMarker('stopmarkers',{R_LQ, R_HQ}, ...
            'bv_bbciclose',0, 'state',state);
        responded = 1;
    end
end





%% Pretest -- Measure perceptual threshold
fprintf('Press <RETURN> to start threshold measurement.\n'); pause;

nTrialsPerCondition = 10;
nConditions = length(vids);
stim_seq = mod(0:nConditions*nTrialsPerCondition-1,nConditions)+1;
stim_seq = stim_seq(randperm(length(stim_seq)));

% Make connection to bv recorder for obtaining trigger data
state= acquire_bv(1000, 'localhost');
state.reconnect= 1;

Resp = [];
QLevel = [];
count = count.reset();
for n = 1:length(stim_seq)
  count = count.oneup();
  vid = vids{stim_seq(n)};
  QLevel = [QLevel int2str(vid(7:end-4))];
  responded = 0;
  % start video
  cmd = ['cd ' VLC_DIR ' & vlc ' VLC_OPTS ' ' videodir vid];
  dos(cmd);
  while ~responded
    s = stimutil_waitForMarker('stopmarkers',{R_LQ, R_HQ}, ...
      'bv_bbciclose',0, 'state',state);
    switch(s)
      case R_HQ
        Resp = [Resp 0];  
        responded = 1;
      case R_LQ 
        Resp = [Resp 1]; 
        responded = 1;  
    end
  end
end
count = count.reset();

% Plot Pretest Results
[SL sort_idx] = sort(stim_seq);
RESP = Resp(sort_idx);
for n = 1:nTrialsPerCondition
    R(n) = mean(RESP((n-1)*nTrialsPerCondition+1:n*nTrialsPerCondition));
end
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]), plot(1:nTrialsPerCondition,R, '.', 'Markersize', 50)
xlabel('\bf tiling level'), ylabel('\bfdetection rate [%]'), title('\bfPretest Results')

[th_deviation th] = min(abs(R-0.5)); % simple/simplistic threshold estimation
% evtl. use also 25% recognized level

LQs =[th-2, th-1, th, th+1];
Stimlevels = [1 LQs 10];

% Save Pretest results
saveFigure_png([TODAY_DIR 'psydata_pretest'])

fprintf('Chosen levels:  %s.\n',num2str(Stimlevels));
save([TODAY_DIR 'pretest'],'Resp','R','th','LQs','Stimlevels')



%% Create Stimulus Sequence for the Experiment & Save it
videos = vids(Stimlevels);  % Use HQ-video and LQ-videos selected in the pretest
N = N_perVid*length(videos);
stim_seq = mod(0:N-1,length(videos))+1;
stim_seq = stim_seq(randperm(length(stim_seq)));
save([TODAY_DIR 'stim_seq'],'stim_seq')


%% Experiment
load([TODAY_DIR 'pretest'])
load([TODAY_DIR 'stim_seq'])
videos = vids(Stimlevels);    % Use HQ-video and LQ-videos selected in the pretest
fprintf('Press key to start the actual experiment... \n')
pause
bvr_startrecording(['videoTelekomLabs' VP_CODE]); pause(0.5)
ppTrigger(RUN_START);

% Make connection to bv recorder for obtaining trigger data
state= acquire_bv(1000, 'localhost');
state.reconnect= 1;

Resp = [];
count = count.reset();
block = 1;
N_perBlock = 75;
for v = (block-1)*N_perBlock+1:N
    count = count.oneup();
    vidname = [videodir videos{stim_seq(v)}];
    ppTrigger(str2num(videos{stim_seq(v)}(7:end-4)));
    % show video
    cmd= ['cd ' VLC_DIR ' & vlc ' VLC_OPTS ' ' vidname];
    dos(cmd);
    % wait for user input
    responded = 0;
    while ~responded
        s = stimutil_waitForMarker('stopmarkers',{R_LQ, R_HQ}, ...
            'bv_bbciclose',0, 'state',state);
        switch(s)
            case R_HQ
                Resp = [Resp 0];
                responded = 1;
            case R_LQ
                Resp = [Resp 1];
                responded = 1;
        end
    end
    pause(t_wait);
    if mod(v,N_perBlock)==0   % long pause every ~10 minutes
        ppTrigger(RUN_END);
        bvr_sendcommand('stoprecording');
        disp([int2str(length(stim_seq)-v) ' videos to go...']);
        keyboard
        if length(stim_seq)-v==0
            break
        end
        bvr_startrecording(['videoTelekomLabs' VP_CODE]);
        ppTrigger(RUN_START); pause(0.5)        
        % Make connection to bv recorder for obtaining trigger data
        state= acquire_bv(1000, 'localhost');
        state.reconnect= 1;
    end
end


% Plot & Save Results
[SL sort_idx] = sort(stim_seq);
RESP = Resp(sort_idx);
R = [];
for n = 1:length(videos)
    R(n) = mean(RESP((n-1)*N_perVid+1:n*N_perVid));
end
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]), plot(Stimlevels,R, '.', 'Markersize', 50)
xlabel('\bf tiling level'), ylabel('\bfdetection rate [%]'), title('\bfExperimental Results')
saveFigure_png([TODAY_DIR 'psydata_exp'])

%%
fprintf('Experiment finished.\n');


%% TODO: Subjective Video Quality Test
startQualitytest(Stimlevels);












