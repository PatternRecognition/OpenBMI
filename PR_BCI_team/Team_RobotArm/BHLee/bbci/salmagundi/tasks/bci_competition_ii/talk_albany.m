file= 'bci_competition_ii/albany_P300_train_avg15';
fig_dir= 'bci_competition_ii/';

[epo, mrk, mnt]= loadProcessedEEG(file);

epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt);

clf; subplot(2,2,[1 2]);
showERP(epo, mnt, {'Fz','FCz','Cz','CPz','Pz'});
xlabel('time [ms]'); 
ylabel('[\muV]');

ival= [310 360];
grid_markIval(ival);
legend;

topo= proc_meanAcrossTime(epo, ival);
topo= proc_average(topo);
subplot(2,2,3);
plotScalpPattern(mnt, topo.x(:,:,1));
title(topo.className{1});

subplot(2,2,4);
plotScalpPattern(mnt, topo.x(:,:,2));
title(topo.className{2});



epo_tsc= proc_t_scale(epo, 0.01);

grid_plot(epo_tsc, mnt);
grid_markRange(epo_tsc.crit, [], 'color',0.4*[1 1 1]);

dscr_ival= [270 360];
grid_markIval(dscr_ival);



epo_rsqu= proc_r_square(epo);
grid_plot(epo_rsqu, mnt);

grid_markIval(dscr_ival);



topo= proc_meanAcrossTime(epo, dscr_ival);
topo= proc_r_square(topo);
clf;
plotScalpPattern(mnt, topo.x(:,:,1));
title(topo.className{1});




fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 5);

clf;
colormap(hot(21));
plot_classification_scalp(fv, mnt, 'LDA');




fv= proc_commonAverageReference(epo_ave);
fv= proc_selectChannels(fv, 'C1','P8');
fv= proc_baseline(fv, [0 150]);
fv= proc_meanAcrossTime(fv, [300 360]);

doXvalidationPlus(fv, 'LDA', [5 10], 3);

C= trainClassifier(fv, 'LDA');

clf;
c1= find(fv.y(1,:));
c2= find(fv.y(2,:));
xx= squeeze(fv.x);
plot(xx(1,c1), xx(2,c1), 'r.');
hold on;
plot(xx(1,c2), xx(2,c2), 'g.');
plotLinBoundary(C.w, C.b)
hold off;
xlabel('potantial at C1');
ylabel('potential at P8');
legend(fv.className);
