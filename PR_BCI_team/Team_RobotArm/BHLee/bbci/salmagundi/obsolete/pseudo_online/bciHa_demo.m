train_file= 'Gabriel_01_10_15/imagGabriel';
train_frac= 0.77;
test_file= train_file;

%FB_TYPE= 'none';
FB_TYPE= 'teletennis';
%FB_TYPE= 'cross2d';
%FB_TYPE= 'traces';

global cnt mrk pp first_test_event time_line goal
[cnt,mrk,mnt]= loadProcessedEEG(test_file);


%% movement discrimination (left vs. right)
global csp_a csp_b
band= [10 15];
[csp_b, csp_a]= butter(7, band/cnt.fs*2);

dscr= struct('ilen', 3000);
dscr.chans= cnt.clab(scalpChannels(cnt));
dscr.proc= ['global csp_a csp_b csp_w; ' ...
            'fv= proc_filt(epo, csp_b, csp_a); ' ...
            'fv= proc_selectIval(fv, 2000, ''end''); ' ...
            '[fv, csp_w]= proc_csp(fv, 1); ' ...
            'fv= proc_variance(fv);'];
dscr.jit= 3000;
dscr.ilen_apply= 2000;
dscr.proc_apply= ['global csp_a csp_b csp_w; ' ...
                  'fv= proc_linearDerivation(epo, csp_w); ' ...
                  'fv= proc_filt(fv, csp_b, csp_a); ' ...
                  'fv= proc_selectIval(fv, 1000, ''end''); ' ...
                  'fv= proc_variance(fv);'];
dscr.model= 'LSR'; dscr.scale= 1;
%dscr.model= 'linearPerceptron';
%dscr.model= 'FisherDiscriminant'; dscr.scale= 50;


feedback_opt= struct('type', FB_TYPE);

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

 case 'teletennis',
  feedback_opt.step= 4;
  feedback_opt.integrate= 1;
  feedback_opt.ival= [-350 0];
  feedback_opt.xLim= [-1 1];
  
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
if isequal(test_file, train_file),
  nTrains= floor(length(mrk.pos)*train_frac);
  test_begin= mrk.pos(nTrains) + train_test_delay/1000*mrk.fs;
else
  nTrains= 0;
  test_begin= max(1, mrk.pos(1) - 2*mrk.fs);
end


[epo, dscr, dscr_wnd]= bciHa_train(train_file, dscr, [], train_frac);
bciDal_prepare_testing;
bciHa_testing;


feedback_tube;

%nFigs= 2;
%nPlots= 4;
%feedback_overview;
