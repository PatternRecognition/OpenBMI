fig_dir= '/home/neuro/blanker/nibbler/Texte/FhG_misc/leitungskram/strategieplanung/kernkompetenz_poster/';
file= {'Guido_04_03_29/imag_lettGuido', 'Guido_04_03_29/imag_moveGuido'};


%% sparse Fisher (SFD)
model_FDlwqx= struct('classy',{{'FDlwqx','*log'}}, 'msDepth',2);
model_FDlwqx.param= [-1:3];

%% linear sparse Fisher (LSFD)
model_FDlwlx= struct('classy',{{'FDlwlx','*log'}}, 'msDepth',2);
model_FDlwlx.param= [-1:3];

%% linear programming machine
model_LPM= struct('classy','LPM', 'msDepth',2, 'std_factor',2);
model_LPM.param= struct('index',2, 'scale','log', ...
                        'value', [-2:2:5]);


[cnt, mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(mrk, 'left','foot');
mt= getElectrodePositions(cnt.clab);
mnt.x= mt.x;
mnt.y= mt.y;

ival= [750 3000];
band= [6 30];

cnt= proc_selectChannels(cnt, 'not','E*');
%cnt= proc_laplace(cnt);
fv= makeEpochs(cnt, mrk, ival);
fv= proc_spectrum(fv, band, kaiser(fv.fs/2,2), fv.fs/4);

model_name= 'LPM';
%model_name= 'FDlwlx';
%model_name= 'FDlwqx';
eval(['model= model_' model_name]);

classy= select_model(fv, model);
C= trainClassifier(fv, classy);

clab_replace= {'Fpz','Fp'; 
               'Fz','F'; 'FCz','FC';
               'Cz','C'; 'CPz','CP';
               'Pz','P'; 'POz','PO'; 'Oz','O'};
ii= chanind(fv, 'not',clab_replace{:,1});
ic= chanind(fv, clab_replace{:,1});
is= strpatternmatch('*1', clab_replace(:,1));
%fv.clab(ii)= '';
yt= ic;
yt(is)= yt(is)+0.5;

hnd= plot_classifierImage(C, fv, 'show_title',0, 'fontSize',10);
set(hnd.ax(1), 'yTick',yt, 'yTickLabel',clab_replace(:,2));
saveFigure([fig_dir 'fs_weight_matrix'], [20 12]);

sz= size(fv.x);
score= reshape(C.w, sz(1:2));
score= squeeze(sqrt(sum(score.^2, 1)));

clf;
map1= cmap_hsv_fade(10, 1/6, [0 1], 1);
map2= cmap_hsv_fade(11, [1/6 0], 1, 1);
colormap([map1; map2(2:end,:)]);
head= mnt_adaptMontage(mnt, fv.clab);
scalpPlot(head, score, 'colAx','range', 'resolution',50, ...
           'shading','interp', 'contour',0);

shiftAxesDown;
saveFigure([fig_dir 'fs_scalp'], [8 6]);


score= reshape(C.w, sz(1:2));
score= squeeze(sqrt(sum(score.^2, 2)));
plot(score, 'LineWidth',3);
xlabel('[Hz]');