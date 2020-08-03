opt_fig= strukt('folder', [TEX_DIR 'presentation/bbci_presentation2010/pics/'], ...
                'format', 'pdf');

% The following value are taken from Fig.3, 2nd row of KruSelFarVauWol08:
acc= [32 50 62 70 74 77 81 84 87 89];
plot(acc, 'LineWidth',2);
set(gca, 'XLim',[0 10], 'YLim',[0 100], 'XGrid','on', 'YGrid','on');
xlabel('# repetitions');
ylabel('symbol selection accuracy  [%]');

set(gca, 'YTick',[0:20:100])
set(gca, 'XTick',[0 5 10])  

printFigure('performance_matrix_KruSelFarVauWol08', [5 10], opt_fig);


load([DATA_DIR 'results/projects/projekt_treder09/cfy_auto'], ...
     'loss','subdir_list','filelist');

acc= 100 - squeeze(loss(:,3,:));
acc_mean= mean(acc);
acc_se= std(acc)/sqrt(size(acc,1));
clf;
h= errorbar(1:10, acc_mean, acc_se, 'Color','m');
hc= get(h,'Children');
set(hc(1), 'LineWidth', 2);
set(gca, 'XLim',[0 10], 'YLim',[0 100], 'XGrid','on', 'YGrid','on');
xlabel('# repetitions');
ylabel('symbol selection accuracy  [%]');

set(gca, 'YTick',[0:20:100])
set(gca, 'XTick',[0 5 10])  

printFigure('performance_hexospell', [5 10], opt_fig);




gug= [12 45.4 24.9 11.7 6];

clf;
xt= [55 65 75 85 95];
hb1= barh(xt+1, gug);
set(hb1, 'FaceColor','b');
set(gca, 'YTick',xt, 'YTickLabel','<60|60-70|70-80|80-90|90-100');
hx= ylabel('[%] hits in feedback');
hy= xlabel('[%] of feedback runs');
set([hx, hy], 'FontWeight','bold');
set(gca, 'YLim',[50 102], 'XGrid','on');

printFigure('hhist_guger', [10 10], opt_fig);


acc_fb= [7 8.5 19.5 37.5 27.5];
clf;
xt= [55 65 75 85 95];
hb2= barh(xt+1, acc_fb);
set(hb2, 'FaceColor',[1 0 1]);
set(gca, 'XTick',0:10:50, 'XGrid','on');
set(gca, 'YTick',xt, 'YTickLabel','<60|60-70|70-80|80-90|90-100');
%hx= ylabel('[%] hits in feedback');
hy= xlabel('[%] of feedback runs');
set([hy], 'FontWeight','bold');
set(gca, 'XLim',[0 50], 'YLim',[50 102], 'XGrid','on');

printFigure('hhist_bbci', [10 10], opt_fig);

