file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1200 600]);
epo= proc_baseline(epo, [-1200 -1000]);
epo= proc_laplace(epo);

chan= chanind(epo, 'C3');
chan2= chanind(epo, 'C4');
tubePercent= [5 10 15];
E= epo.t;
nClasses= size(epo.y,1);
tube= zeros(length(E), 2*length(tubePercent)+3, nClasses);
for it= 1:length(E),
  for ic= 1:nClasses,
    clInd= find(epo.y(ic,:));
%    xx= squeeze(epo.x(it, chan, clInd))';
    xx= squeeze(epo.x(it, chan, clInd)-epo.x(it, chan2, clInd))';
    tube(it, :, ic)= fractileValues(xx, tubePercent);
  end
end

plotTube(tube, E);
%plotTubeNoFaceAlpha(tube, E); %% zu drucken
