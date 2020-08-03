function fb_opt= set_feedback(FB_TYPE)
%feedback_opt= set_feedback(FB_TYPE)

fb_opt= struct('type', FB_TYPE);
fb_opt.combiner_fcn= 'combinerDal';

switch(FB_TYPE),
 case 'cross2d',
  fb_opt.step= 4;
  fb_opt.integrate= 2;
  fb_opt.pause= {0.3};    %% pause in sec (use {} for keywait)

  fb_opt.ival= [-120 0];
  fb_opt.toc= -120;
%  fb_opt.xLim= [-0.03 0.03];
  fb_opt.xLim= [-1.8 1.8];
  fb_opt.yLim= [-1.5 1.5];
  fb_opt.delay= 0.01;     %% slow down animation
  
 case 'traces',
  fb_opt.step= 4;
  fb_opt.integrate= 2;
  fb_opt.pause= {};       %% pause in sec (use {} for keywait)
  fb_opt.stop_delay= 500;
  fb_opt.zoomSec= 6;
  fb_opt.overview= 0;
  fb_opt.overviewSec= 30;
  fb_opt.disp_step= 4;

 case 'none',
  fb_opt.step= 4;
  fb_opt.integrate= 2;
end
