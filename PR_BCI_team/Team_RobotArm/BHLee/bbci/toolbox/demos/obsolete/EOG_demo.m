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


options.lowpass = [0 10];

cnt_flt = proc_removeBlinks(cnt, cntarti, mrkarti, options);
cnt_flt = proc_removeEyemoves(cnt_flt, cntarti, mrkarti, options);

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




