file= 'Gabriel_00_09_05/selfpaced2sGabriel';

FB_TYPE= 'none';
%FB_TYPE= 'cross2d';
%FB_TYPE= 'traces';

global cnt mrk pp first_test_event time_line goal
[cnt,mrk]= loadProcessedEEG(file);
spatFilt= {{'C3 lap', 'C3',1/2, 'C1',1/2, ...
                      'C5',-1/6, 'FC3',-1/6, 'FC1',-1/6, ...
                      'Cz',-1/6, 'CP3',-1/6, 'CP1',-1/6}, ...
           {'C4 lap', 'C4',1/2, 'C2',1/2, ...
                      'Cz',-1/6, 'FC2',-1/6, 'FC4',-1/6, ...
                      'C6',-1/6, 'CP2',-1/6, 'CP4',-1/6}, ...
           {'CP3 lap','CP3',1/2, 'CP1',1/2, ...
                      'CP5',-1/4, 'C3',-1/4, 'C1',-1/4, ...
                      'CPz',-1/4}, ...
           {'CP4 lap', 'CP4',1/2, 'CP2',1/2, ...
                      'CPz',-1/4, 'C2',-1/4, 'C4',-1/4, ...
                      'CP6',-1/4}, ...
           {'Cz lap', 'Cz',1, 'C1',-1/4, 'FCz',-1/4, ...
                      'C2',-1/4, 'CPz',-1/4}, ...
           {'CPz lap','CPz',1, 'CP1',-1/3, 'Cz',-1/3, ...
                      'CP2',-1/3}};
cnt= proc_spatialFilter(cnt, spatFilt);
cnt= proc_baseline(cnt);

%% movement detection / prediction
dtct= [];
dtct.ival= [-1280 0];
dtct.motoJits= [-300 -200 -100 0];
dtct.nomotoJits= [600 800 1000 1200];
dtct.shift= 0;
dtct.chans= {'C3','C4','CP3','CP4','Cz'};
dtct.proc= ['fv= proc_selectIval(epo, 740); ' ...
            'fv= proc_fourierBandReal(fv, [0.8 2.3], 128);'];
dtct.model= 'linearPerceptron';
dtct.scale= 1;


%% movement discrimination (left vs. right)
dscr= [];
dscr.ival= [-1280 0];
dscr.jits= [-250 -180 -90 0];
dscr.chans= {'C3','C4','CP3','CP4'};
dscr.proc= ['fv= proc_selectIval(epo, 740); ' ...
            'fv= proc_fourierBandReal(fv, [0.8 2.3], 128);'];
dscr.scale= 1;
%dscr.model= {'equalpriors', 'LSR'};
dscr.model= 'linearPerceptron';
%dscr.model= 'FisherDiscriminant'; dscr.scale= 50;


feedback_opt= struct('type', FB_TYPE);
feedback_opt.combiner_fcn= 'combinerDal';
feedback_opt.step= 10;
feedback_opt.integrate= 1;

switch(FB_TYPE),
 case 'cross2d',
  feedback_opt.pause= {0.5};    %% pause in sec (use {} for keywait)

  feedback_opt.ival= [-350 0];
  feedback_opt.toc= -120;
%  feedback_opt.xLim= [-0.03 0.03];
  feedback_opt.xLim= [-1.5 1.5];
  feedback_opt.yLim= [-1.5 1.5];
  feedback_opt.delay= 0.01;     %% slow down animation
  
 case 'traces',
  feedback_opt.pause= {};       %% pause in sec (use {} for keywait)
  feedback_opt.stop_delay= 500;
  feedback_opt.zoomSec= 6;
  feedback_opt.overview= 0;
  feedback_opt.overviewSec= 30;
  feedback_opt.disp_step= 4;

 case 'none',
end


train_test_delay= 1220;  %% testing starts XX ms after last training marker
nTrains= 385;

bciJym_train;
bciDal_prepare_testing;

if ~isequal(FB_TYPE, 'none'),
  fprintf('press <ret> to start feedback\n'); pause
end

bciDal_testing;


nFigs= 2;
nPlots= 4;
feedback_overview;
