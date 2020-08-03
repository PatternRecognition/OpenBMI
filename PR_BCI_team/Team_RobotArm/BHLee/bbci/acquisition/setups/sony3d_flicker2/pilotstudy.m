%% Pilotstudie
% Determine flicker perception threshold

%disp('<a href="http://bdml.stanford.edu/twiki/pub/Haptics/DetectionThreshold/psychoacoustics.pdf">Up - Down </a>.')
tic
record=0 % 1: amplifier connected, record EEG data, 0: testing

VP_CODE=input('Enter VP_CODE used in a previous experiment \nin quotation marks and press <ENTER>. \nIf not known: Only press <ENTER>\n');
%acq_makeDataFolder;

% check that VP_CODE is set and TODAY_DIR exists
if isempty(VP_CODE); error('Define VP_CODE!'); end
if isempty(TODAY_DIR); error('Define TODAY_DIR!'); end

% start tcp/ip connection
display('Server started?'),pause
setup_sony3d_flicker2

display('Turn off CRT!'),pause
display('Turn on shutter glasses!'),pause

% for printing figures
global BBCI_PRINTER; BBCI_PRINTER=1;

% Load sounds and images
jungle= imread('jungle_normal','jpg');
[ding, Fs, nbits, readinfo] = wavread('AirPlaneDing');
[bell, Fs2, nbits, readinfo] = wavread('ShipBell');
[Click1, Fs, nbits, readinfo] = wavread('Click1');Click1=Click1(1:1000);

%% Introduction

display_message('Example for strong flicker'),pause(2),closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*39), pause(2)
fullscreen(jungle,2)
pause(4)
closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*99), pause(2)


display_message('Example for flicker turned off'),pause(2),closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*97), pause(2)
fullscreen(jungle,2)
pause(4)
closescreen()


display_message('Example for a medium flicker'),pause(2),closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*47), pause(2)
fullscreen(jungle,2)
pause(4)
closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*99), pause(2)


display_message('What do you think about this?'),pause(2),closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*49), pause(2)
fullscreen(jungle,2)
pause(4)
closescreen()
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640, 480, 2*99), pause(2)

instruction={'When the screen is black, please enter';...
'1 for "flicker perceived" or';... 
'2 for "flicker not perceived"';'';...
'Then press <ENTER>, only the last entry counts';'';...
'Shutter glasses switch between trials.';...
'This is not the flicker.'};

display_message(instruction)
pause,closescreen()

%% Determine perception threshold
step=2; % [Hz]  step_glasses = step --> step_CRT screen = 2*step

images={jungle};
imagenames={'jungle'};

for i=1:numel(images)
  stimulus=images{i};
  
  % Start with a frequency slightly under the perception threshold
  freq=49;
  catchtrial=0;
  lastperception=1;
  results=[];
  
  turns=20; % number of turns from perceived -> not perceived and vice versa
  while turns>0
    
    % Change frequency of CRT screen
    pnet(tcp_conn, 'printf', 'freq %d %d %d\n', 640, 480 ,2*freq);
    fprintf('Shutter glasses frequency: %d\n',freq)
    pause(2)

    % Display stimulus)
    fullscreen(stimulus,2)
    pause(4)
    
    % Black screen
    closescreen()
    
    % Enter whether flicker is perceived (1) or not (2)
    perception = yesno_input();
    soundsc(ding,Fs)
    
    % Save f_glasses, perception, catchtrial(1/0), turn (1/0)
    results=[results; freq perception catchtrial 0]

    % Don't use the catchtrails for the update but the last real trial
    if not(catchtrial)
      turns=turns-(lastperception~=perception);
      results(end,4)=(lastperception~=perception); % turn (1/0)
      lastperception=perception;
      lastfreq=freq;
    end

    % Increase frequency if flicker detected, decrease if not detected. Show 45Hz or 85Hz in 1/4 of the trials
    switch ceil(6*rand)
      case 1
        freq=41; % glasses 45Hz --> CRT 90Hz
        catchtrial=1;
      case 2
        freq=85;% glasses 85Hz --> CRT 190Hz
        catchtrial=1;
      otherwise
        if lastperception == 1
          freq=lastfreq+step;
        else
          freq=lastfreq-step;
        end
        catchtrial=0;
    end
    
  end

  soundsc(bell,Fs2)
  
  % Average frequency, without the first run, without catchtrials
  idx_secondrun=find(results(:,4),1);
  idx_nocatchtrials=find(not(results(:,3)));
  idx_nocatchtrials=idx_nocatchtrials(idx_nocatchtrials>idx_secondrun);
  f_mean=mean(results(idx_nocatchtrials,1));
  
  filename = strcat([TODAY_DIR 'pilot_' VP_CODE,'-',imagenames{i}]);
  save(filename, 'results','f_mean')
  %load(filename, 'results','f_mean')
  
  figure(i),clf
  stairs(results(:,1)), hold on % frequencies  
  plot(1:size(results,1), repmat(f_mean, size(results,1), 1 ), '-r')  % mean frequencies
  stem(0.5+[1:size(results,1)],42*results(:,3),'k') % catchtrials
  stairs(90+results(:,2))% perception 
  set(gca,'YTick',39:step:99)
  xlabel('Trial'); ylabel('Frequency of shutter glasses = f(CRT)/2 [Hz]'),ylim([39 99]);
  title(sprintf(['Perception threshold %0.5g, image %s '], f_mean, imagenames{i}))
  printFigure([filename '_incl_catchtrials'],'maxAspect')
  
  figure(10+i),clf
  stairs(results(idx_nocatchtrials,1)), hold on % frequencies
  plot(1:numel(idx_nocatchtrials), repmat(f_mean, numel(idx_nocatchtrials), 1 ), '-r')  % mean frequencies
  stairs(90+results(idx_nocatchtrials,2))% perception 
  set(gca,'YTick',39:step:99)
  xlabel('Trial'); ylabel('Frequency of shutter glasses = f(CRT)/2 [Hz]'),ylim([39 99]);
  title(sprintf(['Perception threshold %0.5g, image %s '], f_mean, imagenames{i}))
  printFigure(filename,'maxAspect')
  
end


pnet(tcp_conn, 'printf', 'freq %d %d %d\n', 800, 600 ,85);
pnet(tcp_conn, 'close');
fprintf('done!\n');
fprintf('Turn on CRT!\n');
toc
