file= 'bci_competition_ii/albany_P300_train';
fig_dir= 'bci_competition_ii/';

[cnt, mrk, mnt]= loadProcessedEEG(file);

epo= makeEpochs(cnt, mrk, [-50 500]);
epo.code= mrk.code;
epo_ave= proc_albanyAverageP300Trials(epo, 15);

fv= proc_commonAverageReference(epo_ave);
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 5);

clf;
colormap(hot(21));
fv= rmfield(fv, 'title');
plot_classification_scalp(fv, mnt, 'LDA');
title(sprintf('<%s> vs. <%s>', fv.className{:}));
saveFigure([fig_dir 'albany_P300_ave_classy_scalp'], [6 5]*2);



fv= proc_commonAverageReference(epo_ave);
fv= proc_selectChannels(fv, 'C1','P8');
fv= proc_baseline(fv, [0 150]);
fv= proc_meanAcrossTime(fv, [300 360]);

doXvalidationPlus(fv, 'LDA', [5 10], 3);

clf;
c1= find(fv.y(1,:));
c2= find(fv.y(2,:));
xx= squeeze(fv.x);
plot(xx(1,c2), xx(2,c2), 'g.');
hold on;
plot(xx(1,c1), xx(2,c1), 'r.');
C= trainClassifier(fv, 'LDA');
plotLinBoundary(C.w, C.b)
hold off;
xlabel('potantial at C1');
ylabel('potential at P8');
legend(fv.className{2:-1:1});
saveFigure([fig_dir 'albany_P300_distrib'], [10 7]*2);
