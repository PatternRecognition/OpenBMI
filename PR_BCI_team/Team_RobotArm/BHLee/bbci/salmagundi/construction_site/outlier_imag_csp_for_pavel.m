%% ein Beispiel mit 'imagery' Experiment und CSP Klassifikation

%for expno= 1:6, outlier_imag_csp_for_pavel; end

file_list={'Hendrik_02_08_12/imagmultiHendrik', ...
           'Steven_02_08_12/imagmultiSteven', ...
           'Soeren_02_08_13/imagmultiSoeren', ...
           'Gabriel_01_10_15/imagGabriel', ...
           'Thorsten_02_07_31/imagThorsten', ...
           'Ben_02_07_23/imagBen'};
file= file_list{expno};
    
%% Auswahl der Klassen,
%% bei 'imagmulti' Experimenten gibt es noch weitere Klassen
classes= {'left','right'};

%% Parameter zur Vorverarbeitung
csp_band= [7 14];
csp_ival= [500 2500];
csp_nPats= 3;
csp_chans= {'not', 'Fpz','AF#'};

nRestricted= 150;

%% laden der EEG Daten, EMG und EOG Kanäle werden ausgelassen
[cnt,mrk,mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not', 'E*');

%% Marker für die ausgewählten Klassen suchen
clInd= find(ismember(mrk.className, classes));
cli= find(any(mrk.y(clInd,:)));
mrk_cl= pickEvents(mrk, cli);
fprintf('\n%s: <%s> vs <%s>\n', cnt.title, mrk_cl.className{:});

%% Segmente um die Marker ausschneiden
epo= makeSegments(cnt, mrk_cl, [-1500 4000]);
clear cnt

%% Band-pass Filterung
clear cnt

%% Vorverarbeitung für CSP (common spatial patterns)
%% Da ein Teil der Vorverarbeitung klassenabhängig ist, muss sie innerhalb
%% der Kreuzvalidierung auf jeder Trainingsmenge separat gemacht werden
fv= proc_selectChannels(epo, csp_chans);
[b,a]= getButterFixedOrder(csp_band, epo.fs, 6);
fv= proc_filt(fv, b, a);
fv= proc_selectIval(fv, csp_ival);
fv.proc=['fv= proc_csp(epo, ' int2str(csp_nPats) ');' ...
         'fv= proc_variance(fv); '];

%% unregularisierte Klassifikation
fprintf('on all %d trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, 'LDA', [5 10]);

fprintf('on arb %d trials: ', nRestricted); ...
doXvalidationPlus(fv, 'LDA', [5 10 nRestricted]);


%% Artefaktsuche nach Standard Methode
epo_ioi= proc_selectIval(epo, csp_ival);
epo_ioi= proc_selectChannels(epo_ioi, csp_chans);
criteria= struct('gradient',100, 'maxmin',200);
arti_events= find_artifacts(epo_ioi, [], criteria);
fprintf('%d segments marked as EEG-artifacts\n', length(arti_events));

%% mark artifact events as rejected (class 0)
fv.y(:,arti_events)= 0;
fprintf('on all %d artifree trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, 'LDA', [5 10]);

fprintf('on arb %d artifree trials: ', nRestricted); ...
doXvalidationPlus(fv, 'LDA', [5 10 nRestricted]);


%% restore all labels
fv.y= mrk_cl.y;

%% mark events with large variance in interval-of-interest as artifacts
epo_ioi= proc_selectIval(epo, csp_ival);
epo_ioi= proc_selectChannels(epo_ioi, csp_chans);
mcv= squeeze( max(std(epo_ioi.x)) );
arti_events= find(mcv>mean(mcv)+std(mcv));
fprintf('%d segments marked as var-artifacts\n', length(arti_events));

%% mark artifact events as rejected (class 0)
fv.y(:,arti_events)= 0;
fprintf('on all %d artifree trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, 'LDA', [5 10]);

fprintf('on arb %d artifree trials: ', nRestricted); ...
doXvalidationPlus(fv, 'LDA', [5 10 nRestricted]);
