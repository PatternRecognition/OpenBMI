clear all;
file = 'D:\data\bbciRaw\VPiz_08_04_15\fingertip_leftVPiz';
[cnt,mrk]= eegfile_loadBV(file,'fs',100);

miscDef= {102;'start'};
mrk_misc= mrk_defineClasses(mrk, miscDef);

epo = cntToEpo(cnt,mrk_misc,[0 40000]);
dat_fourier = proc_fourierBand(epo,[4 6],1000);
dat= proc_fourierBand(cnt,[4 6],10000);

para.modfreq = [5:2:31];
%plot(dat.t,dat.x(:,1))

for ii = 1:length(para.modfreq)
highfr = para.modfreq(ii)+1;
lowfr = para.modfreq(ii)-1;
dat_energy = proc_fourierBandEnergy(epo, [lowfr highfr], 4000);

for i = 1:59
  mean_dat_en(:,i) = mean(dat_energy.x(:,i,:));
end
matband(:,ii) = mean_dat_en;

sortmean(ii) = max(matband(:,ii));
% ind(:,ii) = find(matband(:,ii) == sortmean(:,1));

subplot(length(para.modfreq),1,ii)
plot(matband(:,ii),'o')
grid on
end


%%
data_cnt = cnt.x(:,1);
f =((0:length(data_cnt)-1)/length(data_cnt))*cnt.fs; 
data_cnt_spec = abs(fft(data_cnt));
plot(f,data_cnt_spec)
xlim([4 30])


