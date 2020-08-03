% Helligkeitswerte:
% PERIPHERES SEHEN: 
% * 28200
% * 1.316
%
% ZENTRALES SEHEN:
%
%

%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%% Experimental design
nTrialsPerCondition = 300;

% Define Stop Markers
marker = {'R  4','R  8','S253'};

% Philips frequency<->code mapping

% *** TO DO: extend frequency range ***

% All possible frequencies and the according codes for the serial port
freqs =  [40:10:70 75 77 80 83 85 87 90 95 100:10:150];
codes =  [26:16:74 75 77 90 83 85 87 106 95 122:16:202];   

vormessungFreqs = [40];
practiceFreqs = [40:10:100]; % Freqs shown during practice
% practiceFreqs = [40:10:60]; % Freqs shown during practice
measureFreqs = [40 50 60 70 75 77 80 83 85 87 90 95 100 110];   % Freqs for threshold measurement

%% Startup pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0)
pause(5)

%% Vormessung : peripheres Sehen
runCodes = codes(ismember(freqs,vormessungFreqs));
nTrialsPerCondition = 100;

fprintf(['Press <RETURN> to start Vormessung [peripher].\n']); pause;
fprintf('Ok, starting...\n'), close all
setup_feedback;
pyff('setint','freq',runCodes);
pyff('setint','nTrialsPerCondition',nTrialsPerCondition);
pyff('setdir','basename','vormessung_peripher');
pause(1)
pyff('play');
pause(5)

% Make connection to bv recorder for obtaining trigger data
state= acquire_bv(1000, 'localhost');
state.reconnect= 1;

while true
  s = stimutil_waitForMarker('stopmarkers',marker,'bv_bbciclose',0, ...
    'state',state);
  switch(s)
    case {'R  4','R  8'}
      pyff('set','waiting',0);
    case 'S253'
      break
  end
end
fprintf('Vormessung [peripher] finished!\n')
pyff('stop');
pyff('quit');

%% --Analyse--

%% Wenn kein peripheres Sehen -> Vormessung : zentrales Sehen
runCodes = codes(ismember(freqs,vormessungFreqs));
nTrialsPerCondition = 150;

fprintf(['Press <RETURN> to start Vormessung [zentral].\n']); pause;
fprintf('Ok, starting...\n'), close all
setup_feedback;
pyff('setint','freq',runCodes);
pyff('setint','nTrialsPerCondition',nTrialsPerCondition);
pyff('setdir','basename','vormessung_zentral');
pause(1)
pyff('play');
pause(5)

% Make connection to bv recorder for obtaining trigger data
state= acquire_bv(1000, 'localhost');
state.reconnect= 1;

while true
  s = stimutil_waitForMarker('stopmarkers',marker,'bv_bbciclose',0, ...
    'state',state);
  switch(s)
    case {'R  4','R  8'}
      pyff('set','waiting',0);
    case 'S253'
      break
  end
end
fprintf('Vormessung [zentral] finished!\n')
pyff('stop');
pyff('quit');

%% Vormessung Analyse
filelist = 'vormessung_peripher';
filelist = 'vormessung_zentral';


%% Vormessung : Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Flickering -- practice
setup_feedback;
pyff('setint','nTrialsPerCondition',2,'doShuffle',0);
% Chose and set frequencies
pyff('setint','freq',codes(ismember(freqs,practiceFreqs)));

fprintf('Press <RETURN> to start practice.\n'); pause;
fprintf('Ok, starting...\n'), close all
pyff('setdir','');   % no recording
pyff('play');
pause(5)

% Make connection to bv recorder for obtaining trigger data
state= acquire_bv(1000, 'localhost');
state.reconnect= 1;
  
while true
  s = stimutil_waitForMarker('stopmarkers',marker,'bv_bbciclose',0, ...
    'state',state);
  switch(s)
    case 'R  4'
      pyff('set','waiting',0);
    case 'R  8'
      pyff('set','waiting',0);
    case 'S253'
      break
  end
end
fprintf('Practice finished!\n')
pyff('stop');
pyff('quit');

%% Measure perceptual threshold using ascending/descending limits
nSweeps = 10;
responses = [];
stimMarker = {};
measureCodes = codes(ismember(freqs,measureFreqs))
for ii=measureCodes
  if ii<100, stimMarker = {stimMarker{:} sprintf('S %d',ii)}; % one whitespace
  else
    stimMarker = {stimMarker{:} sprintf('S%d',ii)}; % no whitespace
  end
end

fprintf('Press <RETURN> to start threshold measurement using %d sweeps.\n',nSweeps); pause;
for ii=1:nSweeps
  downUp = mod(ii,2);  % 0=downward sweep, 1=upward sweep
  setup_feedback;
  nPracticeStim = 4;
  pyff('setint','nTrialsPerCondition',1,'doShuffle',0);
  if downUp==1
    pyff('setint','freq',measureCodes);
  else  % turn around freq codes
    pyff('setint','freq',fliplr(measureCodes));
  end
  fprintf('Ok, starting...\n'), close all
  pyff('setdir','');   % no recording
  pyff('play');
  pause(5)

  % Make connection to bv recorder for obtaining trigger data
  state= acquire_bv(1000, 'localhost');
  state.reconnect= 1;

  lastCode = [];
  marker
  stimMarker
  while true    
    s = stimutil_waitForMarker('stopmarkers',{marker{:} stimMarker{:}}, ...
      'bv_bbciclose',0, 'state',state)
    switch(s)
      case {stimMarker{:}}        
        lastCode = str2num(s(2:end)); %#ok<ST2NM>
      case {'R  4' 'R  8' 'S253'}
        if (downUp && strcmpi(s,'R  4')) || (~downUp && strcmpi(s,'R  8'))
          fprintf('Proceeding with next sweep\n')
          break          
        end        
    end
    pyff('set','waiting',0);

  end
  responses = [responses lastCode]
  
  pyff('stop');
  pyff('quit');
  pause(2)
end

fprintf('Threshold measurement finished!\n')

%% Determine threshold, set aimed frequencies
up = responses(1:2:end); % uneven sweeps are upward
up = freqs(ismember(codes,up));
down = responses(2:2:end); % even sweeps are downward
down = freqs(ismember(codes,down));
fprintf('Upward sweeps: %s Hz\nDownward sweeps: %s Hz\n',num2str(up),num2str(down));

% down=[60 70 80];
% up=[82 86 90]
th = mean(responses)
% Aimed frequencies consists of four points
% 1:10 Hz less than 
% runFreq =[min(down)-10, (min(down)+max(up))/2, max(up), max(up)+20];
runFreq =[th-20, th, th+10, th+20];

% Match runFreq with the available frequencies
sf = sort(freqs);
runFreq(1) = freqs(find(sf>runFreq(1),1,'first')-1); % find next lower frequency
runFreq(2) = freqs(find(sf>runFreq(2),1,'first')-1);
runFreq(3) = freqs(find(sf>runFreq(3),1,'first')+1); % find next upper frequency
runFreq(4) = freqs(find(sf>runFreq(4),1,'first')+1);

fprintf('Chosen frequencies:  %s Hz.\n',num2str(runFreq));
keyboard

%% Flickering -- RUN
runCodes = codes(ismember(freqs,runFreq));
nBlocks = 10;

for ii=1:nBlocks
%%
  fprintf(['Press <RETURN> to start block ' num2str(ii)  '.\n']); pause;
  fprintf('Ok, starting...\n'), close all
  setup_feedback;
  pyff('setint','freq',runCodes);
  pyff('setint','nTrialsPerCondition',nTrialsPerCondition/nBlocks);
  pyff('setdir','basename','philips');
  pause(1)
  pyff('play');
  pause(5)

  % Make connection to bv recorder for obtaining trigger data
  state= acquire_bv(1000, 'localhost');
  state.reconnect= 1;

  while true
    s = stimutil_waitForMarker('stopmarkers',marker,'bv_bbciclose',0, ...
      'state',state);
    switch(s)
      case {'R  4','R  8'}
        pyff('set','waiting',0);
      case 'S253'
        break
    end
  end
  fprintf(['Block ' num2str(ii) ' finished!\n'])
  pyff('stop');
  pyff('quit');
  
%%
end

%%
fprintf('Finished experiment.\n');
