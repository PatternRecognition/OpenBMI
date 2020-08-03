fig_dir= 'nips05_coherence/';

file= 'Guido_04_03_29/imag_lettGuido';
%file= 'Guido_04_03_29/imag_moveGuido';
%file= strcat('Guido_04_03_29/imag_', {'lett','move'}, 'Guido');
%file= strcat('Klaus_04_04_08/imag_', {'lett','move'}, 'Klaus');

opt_plot= struct('colorOrder',[1 0 0;0 0 1]);

[cnt, mrk, mnt]= eegfile_loadMatlab(file);
mrk= mrk_selectClasses(mrk, 'left','foot');
%mrk= mrk_selectClasses(mrk, 'left','right');
grd= sprintf('FC3,FC1,FCz,FC2,FC4\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
mnt= mnt_setGrid(mnt, grd);

epo= makeEpochs(cnt, mrk, [750 3500]);
epo_lap= proc_laplace(epo);

spec= proc_selectChannels(epo_lap, 'FC3-4','C3-4','CP3-4');
spec= proc_spectrum(spec, [5 35]);
grid_plot(spec, mnt, opt_plot, 'xTick',10:10:30);

rs= proc_r_square(spec);
[mm,mi]= max(mean(rs.x,2));
freq= rs.t(mi);
%freq= 12;

ch_list= {'FC3-FCz','FC4-FCz','FC4-FC3', ...
          'C3-Cz','C4-Cz','C4-C3', ...
          'CP3-CPz','CP4-CPz','CP4-CP3', ...
          'FC3-CP3','FCz-CPz','FC4-CP4'};
mnts= getElectrodePositions(ch_list);
grd= sprintf('FC3-FCz,FC4-FCz,FC4-FC3\nC3-Cz,C4-Cz,C4-C3\nCP3-CPz,CP4-CPz,CP4-CP3\nFC3-CP3,FCz-CPz,FC4-CP4');
mnts= mnt_setGrid(mnts, grd);

eph= proc_fourierCourseOfPhase(epo_lap, freq, kaiser(100,2), 5);
eph.clab= strhead(eph.clab);
dph= proc_joinChannels(eph, ch_list);
dph= proc_averageRadians(dph, 'std',1);

for ii= 1:2*length(dph.clab); ddp.x(:,ii)= unwrap(ddp.x(:,ii)); end
grid_plot(dph, mnts, opt_plot, 'yLim',[-pi pi], 'shrinkAxes',[0.95 0.9], ...
          'plotStd',1);




epo= makeEpochs(cnt, mrk, [500 4500]);

fv= epo;
fv= proc_laplace(fv);
fv= proc_fourierCourseOfPhase(fv, freq, kaiser(100,2), 5);
fv.clab= strhead(fv.clab);
fv= proc_joinChannels(fv, ch_list);
fv= proc_meanRadiansAcrossTime(fv);
xvalidation(fv, 'QDA');


ff= proc_selectChannels(fv, 'C4-Cz','FCz-CPz','CP3-CPz','CP4-CP3','FC4-CP4','FC3-CP3');
xvalidation(ff, 'QDA');


plotOverlappingHist(fv, 21, 'chan',{'FC4-CP4'})

xvalidation(fv, 'LDA');
xvalidation(fv, 'QDA');
c1= find(fv.y(1,:));
c2= find(fv.y(2,:));
for cc= 1:length(fv.clab),
  m1= meanRadians(fv.x(1,cc,c1));
  m2= meanRadians(fv.x(1,cc,c2));
  shift= mean([m1 m2]);
  fv.x(1,cc,:)= -pi + mod(fv.x(1,cc,:)+pi-shift, 2*pi);
end
xvalidation(fv, 'LDA');
xvalidation(fv, 'QDA');



epo= makeEpochs(cnt, mrk, [500 5000]);

fv= proc_selectChannels(epo, 'not','E*','Fp*','A*','I*','FAF*','OI*');
fv= proc_fourierCourseOfPhase(fv, freq, kaiser(100,2), 5);

ci1= find(fv.y(1,:));
ci2= find(fv.y(2,:));

nChans= size(fv.x,2);
loss= NaN*zeros(nChans, nChans);
for c1= 1:nChans,
  for c2= setdiff(1:nChans, c1),
    ff= proc_joinChannels(fv, [fv.clab{c1} '-' fv.clab{c2}]);
    ff= proc_meanRadiansAcrossTime(ff);
    m1= meanRadians(ff.x(1,1,ci1));
    m2= meanRadians(ff.x(1,1,ci2));
    shift= mean([m1 m2]);
    if abs(m1-m2)>pi,
      shift= shift + pi;
    end
    ff.x= -pi + mod(ff.x+pi-shift, 2*pi);
    loss(c1, c2)= xvalidation(ff, 'LDA', 'out_prefix',[ff.clab{1} ': ']);
  end
end
save('coherence_matrix','loss','file','clab')

loss= min(loss, 0.5);
imagesc(100*loss);
axis square;
colorbar;
tick_clab= {'Fz','FCz','Cz','CPz','Pz','POz','Oz'};
ctick= chanind(fv, tick_clab);
set(gca, 'YTick',ctick', 'YTickLabel',tick_clab);
%set(gca, 'XTick',ctick', 'XTickLabel',tick_clab);


loss= min(loss, 0.45);
loss_mean= mean(loss);

head= mnt_adaptMontage(mnt, fv);
scalpPlot(head, loss_mean, 'colAx','range');
printFigure([fig_dir 'loss_scalp'], [16 12]);

loss_min= min(loss);
[so,si]= sort(loss_min);
nChans= 42;

opt_scalp= strukt('colAx',[min(loss_min) 0.45], ...
                  'contour', 4, 'contour_lineprop',{'LineWidth',0.5'}, ...
                  'linespec',{'Color',.5*[1 1 1], 'LineWidth',2}, ...
                  'scalePos','none');
opt_text= strukt('HorizontalAli','center', 'VerticalAli','top', ...
                 'FontWeight','bold');
clf;
for ci= 1:nChans,
  cc= si(ci);
  suplot(nChans, ci, 0.02, 0.02);
  Hsp= scalpPlot(head, loss(:,cc), opt_scalp, 'mark_channels',fv.clab(cc));
  ht= text(0, 0.9, fv.clab{cc});
  set(ht, opt_text);
end
printFigure([fig_dir 'loss_scalps'], 'paperSize',[23 16], 'device','png');



