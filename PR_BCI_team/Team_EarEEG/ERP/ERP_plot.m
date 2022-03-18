% 
% rec{2,1} = X(1:200,:,:);
% 
% for i=3:19
%     k= (i-3)*300 + 1 : (i-3)*300 + 300;
%     disp(k(end))
%     rec{i,1} = X(200 + k,:,:);

% end

% %%
% cap_epo_rec{2,4} = cap_epo{2,4};
% cap_epo_rec{2,4}.x = permute(X(1:200,:,:),[3,2,1]);
% cap_epo_rec{2,4}.clab = {'Pz'};
% for i=3:19
%     k= (i-3)*300 + 1 : (i-3)*300 + 300;
% %     disp(k(end))
%     cap_epo_rec{i,4} = cap_epo{i,4};
%     cap_epo_rec{i,4}.x = permute(X(200 + k,:,:),[3,2,1]);
%     cap_epo_rec{i,4}.clab = {'Pz'};
% end

%% plot
% i = 2 * yy1(:,1) + yy1(:,2);
% % standing
% epo0 = proc_selectChannels(cap_epo{2,1},'Pz');
% epo0.x = permute(XX1,[3,2,1]); epo0.y = yy1';
% epo0.event.desc = i;
% 
% ii = 2 * yy4(:,1) + yy4(:,2);
% % walking
% epo1 = proc_selectChannels(cap_epo{2,4},'Pz');
% epo1.x = permute(XX4,[3,2,1]); epo1.y = yy4';
% epo1.event.desc = ii;
% 
% ii = 2 * y(:,1) + y(:,2);
% % reconstruction
% epo2 = proc_selectChannels(cap_epo{2,4},'Pz');
% epo2.x = permute(double(X),[3,2,1]); epo2.y = double(y');
% epo2.event.desc = ii;


epo = epo3;
mnt = cap_mnt{4};

% clab= {'Pz'}; %cap: Pz, ear: L10
% clab = {'O1','O2','Oz','P4','Pz'};
% clab = {'Oz','Pz'};
% clab = {'L10','R8'};
yLimit = [-5 5];
ival=[-200 0; 280 380];
colOrder= [1 0 1; 0.4 0.4 0.4];
% epo = proc_selectChannels(epo, clab);
epo_b= proc_baseline(epo, ref_ival);
epo_r= proc_rSquareSigned(epo);

epo_tar = proc_selectClasses(epo,'target');

epo_ntar = proc_selectClasses(epo,'non-target');

% baseline �Ѱ�
epo_b_tar = proc_selectClasses(epo_b,'target');
epo_b_tar= proc_baseline(epo_b_tar, ref_ival);

epo_b_ntar = proc_selectClasses(epo_b,'non-target');
epo_b_ntar= proc_baseline(epo_b_ntar, ref_ival);

% baseline ���Ѱ�
h1 = figure(1);
epo_plot = epo_tar;
plot(epo_plot.t,squeeze(mean(mean(epo.x,3),2)),...
    'LineWidth',2,'Color', color_m{im})
% ylim([-3.2 4])
h1.Position = [154 750 447 226];

% target
% plot(epo_plot.t,squeeze(mean(mean(epo_ga_tar.x,3),2)),...
%     'LineWidth',2,'Color', color_m{im})
hold on
grid on
% non-target
% plot(epo_plot.t,squeeze(mean(mean(epo_ga_ntar.x,3),2)),...
%     'LineWidth',1,'Color', color_m{im}, 'LineStyle','--')

% baseline �Ѱ�
figure(2)
epo_plot = epo_b_tar;
% target
plot(epo_plot.t,squeeze(mean(mean(epo_b.x,3),2)),...
    'LineWidth',2,'Color', color_m{im})
% hold on
grid on
% non-target
% plot(epo_plot.t,squeeze(mean(mean(epo_b_ntar.x,3),2)),...
%     'LineWidth',1,'Color', color_m{im}, 'LineStyle','--')

% r-square
h3 = figure(3);
plot(epo_plot.t,squeeze(mean(mean(epo_r.x,3),2)),...
    'LineWidth',1,'Color', color_m{im})
hold on
%  ylim([-0.04 0.03])
%  yticks(-0.04:0.01:0.03)
  ylim([-0.01 0.02])
 yticks(-0.01:0.01:0.02)
grid on

h3.Position = [152 100 447 226];
% ylim([-5 5])

% title(EPO{im})

hold off
figure(1)
hold off
figure(2)
hold off
% legend(EPO)
% title(clab)
% h2 = figure;
% H2= plot_scalpEvolutionPlusChannel(epos_r_av, mnt, clab, ival, defopt_scalp_erp, ...
%     'ColorOrder',colOrder,'scalpPlot',false);
% h2.Position = [200 743 560 180];








