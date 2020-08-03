file= 'Gabriel_00_09_05/selfpaced2sGabriel';
[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-120);
epo= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');

k= 0;
for fp= -1300:10:-600,
  for lp= fp+10:10:-500, k=k+1;
    ival(k,:)= [fp lp];
    fprintf('[%5d %5d] ', round([fp lp]));
 
    fv= proc_baseline(epo, [fp lp]);
    fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
    fv= proc_subsampleByMean(fv, 5);

    [errMean(k,:), errStd(k,:)]= ...
      doXvalidation(fv, 'FisherDiscriminant', [10 10], 1);
  end
  fprintf('\n');
end
