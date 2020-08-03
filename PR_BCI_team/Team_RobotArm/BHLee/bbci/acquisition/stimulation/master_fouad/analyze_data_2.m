clc;close all;clear all;
%file = 'D:\data\bbciRaw\VPiz_08_04_15\palm_leftVPiz';
file = 'D:\data\bbciRaw\VPja_08_04_18\FingerTipleftVPja';
[cnt,mrk]= eegfile_loadBV(file,'fs',100);

%%
Para.fs = 22050;
Para.act_time = 2;
Para.ref_time = 2;
Para.modfreq = [17:2:31];
Para.carfreq = 200;

Para.num_trial = 10;
Para.count_dura = 10;
Para.ifi = 1;
Para.num_block = 4;

length_trial =Para.ref_time+(Para.act_time+Para.ifi)*length(Para.modfreq);
chan = 20;
miscDef= {102;'start'};
mrk_misc= mrk_defineClasses(mrk, miscDef);
mnt= getElectrodePositions(cnt.clab);
%%
epo = cntToEpo(cnt,mrk_misc,[0 length_trial*1000]);
epo_fft = proc_fourierBand(epo,[15 40],length(epo.x(:,1)));
figure
grid_plot(epo,mnt)
figure
opt=[];
opt.xUnit='Hz';
opt.scalePolicy='sym';

grid_plot(epo_fft,mnt,opt)
%%
% Visualise all channels and fourierband 4 to 40
dat_fourier = proc_fourierBand(cnt,[15 40],length(cnt.x(:,1)));
figure
imagesc(dat_fourier.t,[1:length(cnt.x(1,:))],abs(dat_fourier.x)','CDataMapping','scaled')
colormap(Hot)
colorbar
%%
ind = chanind(cnt.clab,{'T7','CCP2','CCP3'});

% figure
% plot(dat_fourier.t,abs(dat_fourier.x(:,30)))

%%
for ii = 1:length(Para.modfreq)
highfr = Para.modfreq(ii)+1;
lowfr = Para.modfreq(ii)-1;
dat_energy = proc_fourierBandEnergy(epo, [lowfr highfr], length_trial*100);

  for i = 1:59
    mean_dat_en(:,i) = mean(dat_energy.x(:,i,:));
  end
matband(:,ii) = mean_dat_en;
end
figure
imagesc(Para.modfreq,[1:length(cnt.x(1,:))],matband);colormap(Hot);colorbar;

%%
figure
for kk=1:length(ind)
  subplot(length(ind),1,kk)
plot(Para.modfreq,matband(ind(kk),:),'--rs','LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',10)
title(['Channel ',char(cnt.clab(ind(kk)))],'FontSize',14)
xlabel('Modulation Frequency [Hz]')
ylabel('Frequency Band Energy')
grid on
end
%%
% data_cnt = cnt.x(:,20);
% f =((0:length(data_cnt)-1)/length(data_cnt))*cnt.fs; 
% data_cnt_spec = abs(fft(data_cnt));
% figure
% plot(f,data_cnt_spec)
% xlim([4 30])
% 

