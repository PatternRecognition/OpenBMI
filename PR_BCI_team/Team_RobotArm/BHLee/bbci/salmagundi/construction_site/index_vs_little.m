file= 'Gabriel_01_12_12/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);

classDef= {70, 65, 74, 192; ...
           'left index', 'left little', 'right index', 'right little'};
mrk= makeClassMarkers(mrk, classDef, 1000);
epo= makeSegments(cnt, mrk, [-1300 0]-50);

fv= proc_selectChannels(epo, 'FC5-6', 'CFC5-6', 'C5-6', 'CCP5-6', 'CP5-6');
fv= proc_filtBruteFFT(fv, [0.8 2], 128, 150);
fv= proc_jumpingMeans(fv, 5, 3);

fprintf('four classes> ');
doXvalidation(fv, 'FisherDiscriminant', [5 10]);

fv.y= mrk.y(1:2,:);
fprintf('left: index vs little> ');
doXvalidation(fv, 'FisherDiscriminant', [5 10]);

fv.y= mrk.y(3:4,:);
fprintf('right: index vs little> ');
doXvalidation(fv, 'FisherDiscriminant', [5 10]);

fv.y= [sum(mrk.y(1:2,:)); sum(mrk.y(3:4,:))];
fprintf('left vs right> ');
doXvalidation(fv, 'FisherDiscriminant', [5 10]);


fv= proc_flaten(fv);
nChans= length(fv.clab);
nFeats= size(fv.x,1)/nChans;

fv.y= mrk.y(1:2,:);
C= train_FisherDiscriminant(fv.x, fv.y);
Cw= reshape(C.w, [nFeats nChans]); 
ww= NaN*ones(length(mnt.clab), 1);
for fi= nFeats:-1:1,
  subplot(2,3,fi);
  ww(chanind(mnt.clab, fv.clab))= Cw(fi,:);
  showScalpPattern(mnt, ww, 0, 'none');
end
ylabel('left: index vs little', 'color','k');

fv.y= mrk.y(3:4,:);
C= train_FisherDiscriminant(fv.x, fv.y);
Cw= reshape(C.w, [nFeats nChans]); 
ww= NaN*ones(length(mnt.clab), 1);
for fi= nFeats:-1:1,
  subplot(2,3,3+fi);
  ww(chanind(mnt.clab, fv.clab))= Cw(fi,:);
  showScalpPattern(mnt, ww, 0, 'none');
end
ylabel('right: index vs little', 'color','k');



return


classDef= {[70], [65]; 'left index', 'left little'};
%classDef= {[74], [192]; 'right index', 'right little'};
mrk_il= makeClassMarkers(mrk, classDef, 1000);
epo= makeSegments(cnt, mrk_il, [-1300 0]-50);
