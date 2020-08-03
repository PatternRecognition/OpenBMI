function fb_opt= animate_fb_cross2d(fb_opt, ptr, dtct_out, dscr_out, comb_out)
%fb_opt= animate_fb_cross2d(fb_opt, ptr, dtct_out, dscr_out, comb_out)

%% share variables with feedback_run
global mrk pp first_test_event time_line

%% static variables
persistent dtct_out_ma dscr_out_ma ev

cross_x_perc= 1/20;  %% half width of cross in percent of axis width

if isequal(ptr, 'init'),
  fb_opt.iv= getIvalIndices(fb_opt.ival, fb_opt.fs);
  fb_opt.toc_idx= round(fb_opt.toc*fb_opt.fs/1000);
  dtct_out_ma= zeros(size(dtct_out));
  dscr_out_ma= zeros(size(dtct_out));
  ev= first_test_event;
  
  clf;
  set(gca, 'xLim',fb_opt.xLim, ...
           'yLim',fb_opt.yLim);
  line([xlim; 0 0]', [0 0; ylim]', 'color','k');
  fb_opt.h_schweif= ...
      line(zeros(length(fb_opt.iv),1), ...
           zeros(length(fb_opt.iv),1), 'color',[.5 .5 1]);
  old_units= get(gca, 'units');
  set(gca, 'units','pixel');
  pos= get(gca, 'position');
  set(gca, 'units',old_units);
  cx= diff(xlim)*cross_x_perc;
  cy= pos(3)*cross_x_perc*diff(ylim)/pos(4);
  fb_opt.cross_x= [-cx cx; 0 0]';
  fb_opt.cross_y= [0 0; -cy cy]';
  fb_opt.h_kreuz= ...
      line(fb_opt.cross_x, fb_opt.cross_y, ...
           'color','b', 'lineWidth',3);
  hold on;
  fb_opt.h_toc= plot(0, 0, '.', 'color','b', 'markerSize',18);
  hold off;
  axis off;
  
else
  iv= ptr + fb_opt.iv;
  iv(find(iv<1))= 1;
  p0= max(1, ptr-fb_opt.integrate+1);
  dtct_out_ma(ptr)= mean(dtct_out(p0:ptr));
  dscr_out_ma(ptr)= mean(dscr_out(p0:ptr));
  set(fb_opt.h_schweif, 'xData',dscr_out_ma(iv), 'yData',dtct_out_ma(iv));
  iv= ptr + fb_opt.toc_idx;
  iv(find(iv<1))= 1;
  set(fb_opt.h_toc, 'xData',dscr_out_ma(iv), ...
                    'yData',dtct_out_ma(iv));
  set(fb_opt.h_kreuz(1), ...
      'xData',dscr_out_ma(ptr)+fb_opt.cross_x(:,1), ...
      'yData',dtct_out_ma(ptr)+fb_opt.cross_y(:,1));
  set(fb_opt.h_kreuz(2), ...
      'xData',dscr_out_ma(ptr)+fb_opt.cross_x(:,2), ...
      'yData',dtct_out_ma(ptr)+fb_opt.cross_y(:,2));
  title(sprintf('time: %.1f s', time_line(ptr)));
  drawnow;

  if ev<length(mrk.pos) & pp>=mrk.pos(ev),
    if sign(dscr_out(ptr+fb_opt.toc_idx))==[-1 1]*mrk.y(:,ev),
      col= [0 .75 .1];
    else
      col= 'r';
    end
    set(fb_opt.h_toc, 'color',col);
    pause(fb_opt.pause{:});
    set(fb_opt.h_toc, 'color','b');
    ev= ev+1;
    waitForSync;
  else
    pause(fb_opt.delay);
  end
  
end
