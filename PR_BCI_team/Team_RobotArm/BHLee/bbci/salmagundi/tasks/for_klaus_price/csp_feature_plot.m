fig_dir= 'preliminary/';


file= 'New/keuntae_2012_11_09_1_new';
%file= 'Matthias_06_02_09/imag_lettMatthias';
[sbj, datestr]= expbase_decomposeFilename(file)
[cnt,mrk]= eegfile_loadMatlab(file);
bbci= eegfile_loadMatlab([sbj '_' datestr '/imag_1drfb' sbj],'vars','bbci');
cnt= proc_selectChannels(cnt, bbci.setup_opts.clab);
[b,a]= butter(5, bbci.setup_opts.band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);
mrk= mrk_selectClasses(mrk, bbci.classes)
fv= makeEpochs(cnt_flt, mrk, bbci.setup_opts.ival);
[fv, csp_w, csp_eig, csp_a]= proc_csp3(fv, 1);
fv= proc_variance(fv);
fv= proc_flaten(fv);
ci1= find(fv.y(1,:));
ci2= find(fv.y(2,:));
clf;
hp1= plot(fv.x(1,ci1), fv.x(2,ci1), 'ro');
hold on;
hp2= plot(fv.x(1,ci2), fv.x(2,ci2), 'gx');
set(hp2, 'Color',[0 0.7 0]);
set([hp1 hp2], 'LineWidth',1);
hold off;
legend('left','right');
xlabel('var(CSP_L)');
ylabel('var(CSP_R)');
printFigure([fig_dir 'csp_feature_plot'], [12 12]);

pause

fv= proc_logarithm(fv);
clf;
hp1= plot(fv.x(1,ci1), fv.x(2,ci1), 'ro');
hold on;
hp2= plot(fv.x(1,ci2), fv.x(2,ci2), 'gx');
set(hp2, 'Color',[0 0.7 0]);
set([hp1 hp2], 'LineWidth',1);
hold off;
legend(fv.className);
xlabel('log(var(CSP_L))');
ylabel('log(var(CSP_R))');
set(gca, 'XTick',-3:1, 'YTick',-3:1);
printFigure([fig_dir 'csp_log_feature_plot'], [12 12]);

pause

set(gca, 'XLimMode','manual', 'YLimMode','manual');
C= trainClassifier(fv, 'FD');
az= atan(-C.w(2)/C.w(1));
rot= [cos(az) sin(az); -sin(az) cos(az)];
bn= C.b/sqrt(C.w'*C.w);              %% normalized bias
Fr= rot * [bn bn; -200 200];
hold on;
hp= plot(Fr(1,:), Fr(2,:), 'k', 'lineWidth',2);
hold off;
printFigure([fig_dir 'csp_classy'], [12 12]);
