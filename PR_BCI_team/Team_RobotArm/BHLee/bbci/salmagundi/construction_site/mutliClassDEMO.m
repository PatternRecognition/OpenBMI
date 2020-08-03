file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-120);
epo_no = makeSegments(cnt,mrk, [-1300 0]-1000);

epo_no.y = ones(1,size(epo_no.y,2));
epo_no.t = epo.t;
epo_no.className = {'noMovement'};

epo = proc_appendEpochs(epo,epo_no);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);

model = {'multiClass',[1 0;0 1;-1 -1],inline('-x','x'),'LSR'};

fv2= proc_flaten(fv);
C = train_multiClass(fv2.x,fv2.y,[1 0;0 1;-1 -1],inline('-x','x'), 'LSR');
out = apply_multiClass(C,fv2.x);

% in der ersten Zeile stehen nun die Tests links gegen nomovement,
% (positiv yu links), in der zweiten Zeile rechts gegen nomovement,
% die letzte Zeile enthaelt die Summe der beiden anderen (mit
% negiertem Vorzeichen). Positive Zahlen tendieren also zur Klassen
% hin!!!

    


doXvalidationPlus(fv,model,[10 10]);
% zugegebenermassen sehr schlecht, aber es ging hier ja auch nicht
% ums preprocessing!!!
