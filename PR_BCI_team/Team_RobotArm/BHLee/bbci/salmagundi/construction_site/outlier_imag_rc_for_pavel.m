%% ein Beispiel mit 'imagery' Experiment und RC Klassifikation

%for expno= 1:6, outlier_imag_rc_for_pavel; end

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
ar_band= [4 35];
ar_ival= [500 2500];
ar_order= 6;
ar_chans= {'FC3-4','C3-4','CP3-4','P3-4','PO#','O#'};

%% try classification also on sets of restricted size
nRestricted= 150;
%% show model selection results (1) or not (0)
msShow= 0;

%% laden der EEG Daten, EMG und EOG Kanäle werden ausgelassen
[cnt,mrk,mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not', 'E*');

%% Marker für die ausgewählten Klassen suchen
clInd= find(ismember(mrk.className, classes));
cli= find(any(mrk.y(clInd,:)));
mrk_cl= pickEvents(mrk, cli);
fprintf('%s: <%s> vs <%s>\n', cnt.title, mrk_cl.className{:});

%% Segmente um die Marker ausschneiden
epo= makeSegments(cnt, mrk_cl, [-1500 4000]);
clear cnt

%% Vorverarbeitung für RC (reflection coefficients)
[b,a]= getButterFixedOrder(ar_band, epo.fs, 6);
fv= proc_filt(epo, b, a);
fv= proc_laplace(fv, 'small', ' lap', 'filter all');
fv= proc_selectChannels(fv, ar_chans);
fv= proc_selectIval(fv, ar_ival);
fv= proc_rcCoefsPlusVar(fv, ar_order);

%% regularisierte Klassifikation
model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.001 0.01 0.1 0.5];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))], msShow);
fprintf('on all %d trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, classy, [5 10]);

classy= selectModel(fv, model, [3 10 round(9/10*nRestricted)], msShow);
fprintf('on arbitrary %d trials: ', nRestricted); ...
doXvalidationPlus(fv, classy, [5 10 nRestricted]);


%% Artefaktsuche nach Standard Methode
criteria= struct('gradient',100, 'maxmin',200);
epo_ioi= proc_selectIval(epo, ar_ival);
epo_ioi= proc_selectChannels(epo_ioi, ar_chans);
arti_events= find_artifacts(epo_ioi, [], criteria);
fprintf('%d segments marked as EEG-artifacts\n', length(arti_events));

%% mark artifact events as rejected (class 0)
fv.y(:,arti_events)= 0;
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))], msShow);
fprintf('on all %d artifact free trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, classy, [5 10]);

classy= selectModel(fv, model, [3 10 round(9/10*nRestricted)], msShow);
fprintf('on arbitrary %d artifact free trials: ', nRestricted); ...
doXvalidationPlus(fv, classy, [5 10 nRestricted]);


%% restore all labels
fv.y= mrk_cl.y;

%% mark events with large variance in interval-of-interest as artifacts
epo_ioi= proc_selectIval(epo, ar_ival);
epo_ioi= proc_selectChannels(epo_ioi, ar_chans);
mcv= squeeze( max(std(epo_ioi.x)) );
arti_events= find(mcv>mean(mcv)+std(mcv));
fprintf('%d segments marked as var-artifacts\n', length(arti_events));

%% mark artifact events as rejected (class 0)
fv.y(:,arti_events)= 0;
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))], msShow);
fprintf('on all %d artifact free trials: ', sum(any(fv.y))); ...
doXvalidationPlus(fv, classy, [5 10]);

classy= selectModel(fv, model, [3 10 round(9/10*nRestricted)], msShow);
fprintf('on arbitrary %d artifact free trials: ', nRestricted); ...
doXvalidationPlus(fv, classy, [5 10 nRestricted]);
