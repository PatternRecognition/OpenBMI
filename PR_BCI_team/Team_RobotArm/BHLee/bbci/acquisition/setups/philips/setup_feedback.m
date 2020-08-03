fb= struct();
fbint= struct();   % Integers

% fbint.freq = [26,42,58,74,90,106,122,138,154,170,186,202]; % Kodierung fuer seriellen Port
fbint.duty = [0,0];
fbint.nTrialsPerCondition = nTrialsPerCondition;
fbint.doShuffle = 1;

%% Timing
fb.tPreTrial = .8;
fb.tPreStim = .2;
fb.tStim = 2.5;  % was 2 for first 2 subjects
fb.tBetweenStim = 2;
fb.tAfterTrial = 3;
fb.task = 'yes-no';   
% fb.task = '2IFC';   
fb.propCW = .14;   % Proportion of CW trials (for yes/no task)

%% Brightness
% % Settings fuer peripher
% % fbint.brightness1 = 11;
% % fbint.brightness_cw1 = 11;
% fbint.brightness_cw2 = 180;
% fbint.brightness_2 = 180;
% 
% 
% % Settings fuer zentral
% fbint.brightness1 = 1;
% fbint.brightness_cw1 = 1;
% fb.brightness_cw2 = 80;
% fb.brightness_2 = 80;

%% Init Speller and send settings
pyff('init','DetectionTaskPhilips');
pyff('set',fb);
pyff('setint',fbint);