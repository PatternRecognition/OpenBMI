file= 'Pavel_01_11_23/selfpaced2sPavel'; 
arti = 'Pavel_01_11_23/artePavel'; 

[cnt, mrk, mnt]= loadProcessedEEG(file);
[cntarti,mrkarti,mntarti] = loadProcessedEEG(arti); 

% it is good first to make baselinecorrection of the calibration
% data. Therefore search for ranges where nothing happened
iv = get_relevantArtifacts(mrkarti,'Augen offen & entspannen');
interval = iv.int{1};
% at the edges of the command there are maybe some movements,
% therefore make the interval less
interval = interval*[0.9 0.1; 0.1 0.9];
cntarti = proc_baseline(cntarti,interval);


% the call of proc_removeBlinks should be done one the whole EEG
% there are some options, but a lot of defaults, here the defaults
% with some explanations

options.range = [-300 300]; 
% 0 is always the mean of the blink (the highest point), here the
% range between 300 ms before and after this point is considered

options.channels = []; 
% this means that the artifact reduction only see for this
% channels. cnt will be given back only for this channels. (NOTE:
% all channels are used for finding places of artifacts in artifact
% mesasurement (EOGv is a good place to find good examples of
% blinks), but after this only the given channels are used.) [] use
% all channels

options.name = 'blinzeln';  
% the name of the artifact in the artifact measurement

options.method = 'readSamples_blinks';
% a method (function) to find samples given the calibration datas
% and the intervals where the artifacts are.Tthere are three
% arguments for the call
% 1. cntarti
% 2. intervals
% 3. struct with the following entries
%    range: see above
%    further fields: see below, options.methodoptions

options.methodoptions = [];   
% options for options.method

% In the case readSamples_blinks following entries are possible
% (the default are called)

options.methodoptions.number = []; 
% a maximal number of artifacts in the interval ([] so much as possible)

options.methodoptions.channels = 'EOGv'; 
% a channel where the program should see for suitable points of
% blinks, it is not necessary to have this channel in
% options.channels	

options.methodoptions.maxpeak = 1000; 
% a blink must go over 1000 mikroV in the given channel

options.methodoptions.nopeak = 200; 
% the EEG must go under this point to come to an next artifact
 
options.methodoptions.absolut = 1; 
% absolut values of the EEG are taken, 0 real values are taken

options.lowpass = [];   
% lowpassfiltering of the artifact datas with proc_filtByFFT, []
% means no filtering

options.number = 1; 
% the number of SourceVectors the program should calculate

% now the call of the artifactreduction 
%cnt = proc_removeBlinks(cnt, cntarti, mrkarti, options);
% the same meaning has:
% cnt = proc_removeBlinks(cnt, cntarti, mrkarti)   or
% cnt = proc_removeBlinks(cnt, cntarti, mrkarti, [-300 300])


% to get nicer curves (if you want to plot the SOurceWaveforms)
options.lowpass = [0 10];
[cnt_flt,values] = proc_removeBlinks(cnt, cntarti, mrkarti, options);

%values is a struct with the following entries
% variance: proportion of the variance of the ocular potential in
% the artifact mesasuremt
% SV: the SOurceVectors
% SW: the SourceWaveforms

% no everything goes on like every time

%maybe so (classification demo) (here for uncorrected data)
epo= makeSegments(cnt, mrk, [-1300 0]-120);

nTrials= [10 10];               %% for 10 times 10-fold cross-validation
msTrials= [3 10 round(9/10*size(epo.y,2))];       %% for model selection

%% do some preprocessing
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_subsampleByMean(fv, 5);

%% cross-validation for preprocessed data with Fisher Discriminant
%without correction
doXvalidation(fv, 'FisherDiscriminant', nTrials);

% and the same for the corrected data
epo= makeSegments(cnt_flt, mrk, [-1300 0]-120);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_subsampleByMean(fv, 5);

%% cross-validation for preprocessed data with Fisher Discriminant
%with correction
doXvalidation(fv, 'FisherDiscriminant', nTrials);













