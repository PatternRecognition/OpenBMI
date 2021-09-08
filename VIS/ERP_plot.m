h1 = figure; 
% h1.Position = [200 743 560 180];
EPO = {'cap_epo','cap_epo_filt','cap_epo_ASR','cap_epo_rASR','cap_epo_dFFT',...
    'cap_epo_IMU_cICA_PCA_AF','cap_epo_IMU_cICA_PCA_AF2',...
    'cap_epo_i_cICA_PCA_AF','cap_epo_i_cICA_PCA_AF2','cap_epo_i_cICA_PCA_AF2'};

% EPO = {'ear_epo','ear_epo_filt','ear_epo_ASR','ear_epo_rASR','ear_epo_dFFT',...
%     'ear_epo_IMU_cICA_PCA_AF','ear_epo_IMU_cICA_PCA_AF2',...
%     'ear_epo_i_cICA_PCA_AF','ear_epo_i_cICA_PCA_AF2','ear_epo_i_cICA_PCA_AF2'};
%% ERP P300 plot
% sub = 12;
ispeed = 4;
epo_ga = [];
epo_r_ga=[];
erp_sub5 = [6, 1:2, 9:19];
erp_sub = [6 1:5 7:19];
for im = 1:length(EPO)
for sub = erp_sub %1:19
eval(sprintf('epo = %s{sub,ispeed};',EPO{im}));
mnt = cap_mnt{sub};

clab= {'Pz'}; %cap: Pz, ear: L10
yLimit = [-5 5];
ival=[-100 0; 150 200; 200 300];
colOrder= [1 0 1; 0.4 0.4 0.4];
epo = proc_selectChannels(epo, clab);
epo= proc_baseline(epo, ref_ival);
epo_r= proc_rSquareSigned(epo);
if sub == 6
    epo_ga = epo;
    epo_r_ga = epo_r;
else
    epo_ga = proc_appendEpochs(epo_ga, epo);
    epo_r_ga = proc_appendEpochs(epo_r_ga, epo_r);
end
end

epos_av = proc_grandAverage(epo_ga, 'Average', 'INVVARweighted', 'Stats', 1, 'Bonferroni', 1, 'Alphalevel', 0.01);

epos_r_av = proc_grandAverage(epo_r_ga, 'Average', 'INVVARweighted', 'Stats', 1, 'Bonferroni', 1, 'Alphalevel', 0.01);

h1 = subplot(5,2,im);
H1= plot_scalpEvolutionPlusChannel(epos_r_av, mnt, clab, ival, defopt_scalp_erp, ...
    'ColorOrder',colOrder,'scalpPlot',false,'subplot',h1);
ylim([-0.015 0.02])
% ylim([-5 5])

title(EPO{im})
end
title(clab)
% h2 = figure;
% H2= plot_scalpEvolutionPlusChannel(epos_r_av, mnt, clab, ival, defopt_scalp_erp, ...
%     'ColorOrder',colOrder,'scalpPlot',false);
% h2.Position = [200 743 560 180];
%%
% sub = 4;
% ispeed = 4;
% 
% epo = cap_epo{sub,ispeed};
% % epo = cap_epo_i_cICA_PCA_AF{sub,ispeed};
% mnt = cap_mnt{sub};
% h = figure;
% clab= {'Pz'};
% yLimit = [-10 10];
% % epo_r= proc_rSquareSigned(epo);
% epo_r= proc_baseline(epo, ref_ival);
% % plot_channel(epo_r, clab,'Legend',1);
% ylim(yLimit)
% h.Position = [680 743 560 235];
% %%
% 
% % epo_r= proc_rSquareSigned(epo);
% epo_r= proc_baseline(epo, ref_ival);
% % subplot(length(big_epo),1,i)
% plot_channel(epo_r, clab,'Legend',1);
% ylim(yLimit)

