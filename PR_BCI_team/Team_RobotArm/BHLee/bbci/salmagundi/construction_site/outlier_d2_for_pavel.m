%% ein Beispiel mit 'd2' Experiment

%for expno= 1:8, outlier_d2_for_pavel; end

% time-point of classification [ms relative to keypress]
toc= -80;
% interval to check for artifacts
mrp_ival= [-800 toc];

expbase= readDatabase;
%nDtwos= getExperimentIndex(expbase, 'dzwei');  %% expno in 1..nDtwos
idx= getExperimentIndex(expbase, 'dzwei', [], expno);

%% laden der EEG Daten, EMG und EOG Kanäle werden ausgelassen
[cnt, mrk, mnt]= loadProcessedEEG(expbase(idx).file);
cnt= proc_selectChannels(cnt, 'not', 'E*');
fprintf('\nclassifying d2 data of %s at %d ms\n', expbase(idx).subject, toc);

%% Segmentbildung und Vorverarbeitung
epo= makeSegments(cnt, mrk, [-1270 0] + toc);
epo= proc_selectChannels(epo, 'F#','FC#','C#','CP#');
fv= proc_filtBruteFFT(epo, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);

%% try classification also on sets of restricted size
nRestricted= 150;
%% show model selection results (1) or not (0)
msShow= 0;
nTrials= [10 10];

%% unregularisierte Klassifikation
fprintf('on all %d trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, 'LDA', nTrials);

fprintf('on arb %d trials: ', nRestricted); ...
doXvalidationPlus(fv, 'LDA', [nTrials nRestricted]);


%% Artefaktsuche nach Standard Methode
epo_ioi= proc_selectIval(epo, mrp_ival);
criteria= struct('gradient',60, 'maxmin',150);
arti_events= find_artifacts(epo_ioi, [], criteria);
fprintf('%d segments marked as EEG-artifacts\n', length(arti_events));

%% mark artifact events as rejected (class 0)
fv.y(:,arti_events)= 0;
fprintf('on all %d artifree trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, 'LDA', nTrials);

fprintf('on arb %d artifree trials: ', nRestricted); ...
doXvalidationPlus(fv, 'LDA', [nTrials nRestricted]);


%% restore all labels
fv.y= mrk.y;

%% mark events with large variance in interval-of-interest as artifacts
epo_ioi= proc_selectIval(epo, mrp_ival);
mcv= squeeze( max(std(epo_ioi.x)) );
arti_events= find(mcv>mean(mcv)+std(mcv));
fprintf('%d segments marked as var-artifacts\n', length(arti_events));

%% mark artifact events as rejected (class 0)
fv.y(:,arti_events)= 0;
fprintf('on all %d artifree trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, 'LDA', nTrials);

fprintf('on arb %d artifree trials: ', nRestricted); ...
doXvalidationPlus(fv, 'LDA', [nTrials nRestricted]);
