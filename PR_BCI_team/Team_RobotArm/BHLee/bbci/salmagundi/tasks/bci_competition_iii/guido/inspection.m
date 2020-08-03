su = 1; % SUBJECT NUMBER

% GET THE DATA
file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/'];


[cnt,mrk,mnt] = loadProcessedEEG([file_dir 'data_set_v_' char(su+64)]);


% GENERAL SETUP
band = [7 15; 18 26];
bandname = {'alpha','beta'};

nBands = size(band,1);



% CALCULATE THE SPECTRA
%% spectra
min_len= min(diff(mrk.pos))/mrk.fs*1000;
epo= makeEpochs(cnt, mrk, [0 min_len]);
spec= proc_spectrum(epo, [1 105]);
grid_plot(spec, mnt);

spec= proc_spectrum(epo, [1 35]);
grid_plot(spec, mnt);


%CLASS TOPOGRAPHIES
for ib = 1:nBands;
  spec1 = proc_spectrum(epo,band(ib,:),spec.fs);
  spec1 = proc_classMean(spec1,1:length(spec1.className));
  
  spec1.x = spec1.x - repmat(mean(spec1.x,3),[1,1,size(spec1.x,3)]);
  spec1 = mean(spec1.x,1);
    
  clf;
  for iiii = 1:size(spec1,3)
    subplot(1,size(spec1,3),iiii);
    if iiii==size(spec1,3)
      plotScalpPattern(setElectrodeMontage(spec.clab),spec1(:,:,iiii),struct('scalePos','vert','colAx','sym'));
    else
      plotScalpPattern(setElectrodeMontage(spec.clab),spec1(:,:,iiii),struct('scalePos','none','colAx','sym'));
    end
  end
  addTitle([sprintf('%s, [%d %d]-Hz spectral differences',untex(cnt.title),band(ib,1),band(ib,2)),sprintf(', %s',spec.className{:})]);
  
end



% R^2-values
spec_rsq= proc_r_square(spec);
grid_plot(spec_rsq, mnt);


% R^2-Topographien
clf;
for ib= 1:nBands,
  topo= proc_meanAcrossTime(spec_rsq, band(ib,:));
  for j = 1:size(topo.x,3)
    subplot(1,size(topo.x,3),j);
    plotScalpPattern(setElectrodeMontage(spec_rsq.clab), topo.x(:,:,j), scalp_opt);
    title(sprintf('%s  [%d-%d Hz]', topo.className{j}, band(ib,:)));
  end
  addTitle(untex(topo.title), 1, 0);
end



% ERD (power over time) + r^2
for ba = 1:size(band,1)
[b,a] = butter(5,band(ba,:)/cnt.fs*2);
cnt_flt = proc_filt(cnt,b,a);
epo = makeEpochs(cnt_flt,mrk,[500 min_len]);
epo = proc_rectifyChannels(epo);
epo = proc_baseline(epo,250);
epo = proc_movingAverage(epo,200);
grid_plot(epo,mnt);

epo_rsq= proc_r_square(epo);
grid_plot(epo_rsq, mnt);

end





%% evoked potentials
cnt_flt = proc_movingAverage(cnt,20);
epo_slow= makeEpochs(cnt_flt, mrk, [-250 min_len]);
epo_slow= proc_baseline(epo_slow, [-250 0]);
grid_plot(epo_slow, mnt);



% TOPOGRAPHIES
epo_s = proc_classMean(epo_slow,1:length(epo_slow.className));

epo_s = mean(epo_s.x,1);
    
clf;
for iiii = 1:size(epo_s,3)
  subplot(1,size(epo_s,3),iiii);
  if iiii==size(epo_s,3)
    plotScalpPattern(setElectrodeMontage(epo.clab),epo_s(:,:,iiii),struct('scalePos','vert','colAx','sym'));
  else
    plotScalpPattern(setElectrodeMontage(epo.clab),epo_s(:,:,iiii),struct('scalePos','none','colAx','sym'));
  end
end
addTitle([sprintf('%s',untex(cnt.title)),sprintf(', %s',epo.className{:})]);



% r^2-Values
epo_rsq= proc_r_square(epo_slow);
grid_plot(epo_rsq, mnt);

% r^2 Values as topographies
clf;
topo= proc_meanAcrossTime(epo_rsq, [0 min_len]);
for j = 1:size(topo.x,3)
  subplot(1,size(topo.x,3),j);
  plotScalpPattern(setElectrodeMontage(epo_rsq.clab), topo.x(:,:,j), scalp_opt);
  title(topo.className{j});
end
addTitle(untex(topo.title), 1, 0);





