train_file= 'Gabriel_00_09_05/selfpaced2sGabriel';
train_frac= 3/4;
test_file= train_file;

%FB_TYPE= 'none';
FB_TYPE= 'cross2d';
%FB_TYPE= 'traces';

global cnt mrk pp first_test_event time_line goal
[cnt,mrk]= loadProcessedEEG(test_file);

%% movement detection / prediction
dtct= [];
dtct.ival= [-1280 0];
dtct.motoJits= [-70, -100, -130, -160, -200];
dtct.nomotoJits= [-1000, -1100, -1200, 550, 750];
dtct.shift= 0;
dtct.chans= {'FC#', 'C#', 'CP#'};
dtct.proc= ['fv= proc_filtBruteFFT(epo, [0.8 2.3], 128, 300); ' ...
            'fv= proc_jumpingMeans(fv, 6);'];
dtct.scale= 1;
dtct.model= 'LSR';


%% movement discrimination (left vs. right)
dscr= [];
dscr.ival= [-1280 0];
dscr.jits= [-50, -100, -150];
dscr.chans= {'FC#', 'C#', 'CP#'};
dscr.proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
            'fv= proc_jumpingMeans(fv, 5);'];
dscr.scale= 1;
dscr.model= 'LSR';
%dscr.model= 'linearPerceptron';
%dscr.model= 'FisherDiscriminant'; dscr.scale= 50;


feedback_opt= struct('type', FB_TYPE);
feedback_opt.combiner_fcn= 'combinerDal';

switch(FB_TYPE),
 case 'cross2d',
  feedback_opt.step= 4;
  feedback_opt.integrate= 2;
  feedback_opt.pause= {0.5};    %% pause in sec (use {} for keywait)

  feedback_opt.ival= [-350 0];
  feedback_opt.toc= -120;
%  feedback_opt.xLim= [-0.03 0.03];
  feedback_opt.xLim= [-1.8 1.8];
  feedback_opt.yLim= [-1.5 1.5];
  feedback_opt.delay= 0.01;     %% slow down animation
  
 case 'traces',
  feedback_opt.step= 4;
  feedback_opt.integrate= 2;
  feedback_opt.pause= {};       %% pause in sec (use {} for keywait)
  feedback_opt.stop_delay= 500;
  feedback_opt.zoomSec= 6;
  feedback_opt.overview= 0;
  feedback_opt.overviewSec= 30;
  feedback_opt.disp_step= 4;

 case 'none',
  feedback_opt.step= 4;
  feedback_opt.integrate= 2;
end


train_test_delay= 250;  %% testing starts XX ms after last training marker

[epo, dtct, dscr, dtct_wnd, dscr_wnd]= ...
    bciDal_train(train_file, dtct, dscr, train_frac);
if isequal(test_file, train_file),
  nTrains= floor(length(mrk.pos)*train_frac);
  test_begin= mrk.pos(nTrains) + train_test_delay/1000*mrk.fs;
else
  nTrains= 0;
  test_begin= max(1, mrk.pos(1) - 2*mrk.fs);
end

bciDal_prepare_testing;

if ~isequal(FB_TYPE, 'none'),
  fprintf('press <ret> to start feedback\n'); pause
end

bciDal_testing;


nFigs= 2;
nPlots= 4;
feedback_overview;
