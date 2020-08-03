function fb_opt= animate_fb_cross2d(fb_opt, dtct_out, dscr_out, ptr)
%fb_opt= animate_fb_cross2d(fb_opt, dtct_out, dscr_out, ptr)

%% share variables with feedback_run
global cnt mrk pp evt


cross_x_perc= 1/20;  %% half width of cross in percent of axis width

if nargin==1,
  iv_end= ceil(fb_opt.ival(end)*fb_opt.fs/1000);
  iv_len= 1+ceil(diff(fb_opt.ival)*fb_opt.fs/1000);
  iv_intlen= ceil(iv_len/fb_opt.integrate);
  iv_len= fb_opt.integrate*iv_intlen;
  fb_opt.iv= iv_end-iv_len+1:iv_end;
  fb_opt.iv_shape= [fb_opt.integrate iv_intlen];
  toc_idx= getIvalIndices(fb_opt.toc, fb_opt.fs);
  fb_opt.toc_intidx= iv_intlen + ...
                     round( (toc_idx - iv_end) / fb_opt.integrate );

  clf;
  set(gca, 'xLim',fb_opt.xLim, ...
           'yLim',fb_opt.yLim);
  line([xlim; 0 0]', [0 0; ylim]', 'color','k');
  fb_opt.h_schweif= ...
      line(zeros(size(fb_opt.iv_shape,2),1), ...
           zeros(size(fb_opt.iv_shape,2),1), 'color',[.5 .5 1]);
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
  if mod(ptr, fb_opt.integrate)==0,
    iv= ptr + fb_opt.iv;
    iv(find(iv<1))= 1;
    dtct_int= mean(reshape(dtct_out(iv), fb_opt.iv_shape), 1);
    dscr_int= mean(reshape(dscr_out(iv), fb_opt.iv_shape), 1);
    keyboard
    set(fb_opt.h_schweif, 'xData',dscr_int, 'yData',dtct_int);
    set(fb_opt.h_toc, 'xData',dscr_int(fb_opt.toc_intidx), ...
                      'yData',dtct_int(fb_opt.toc_intidx));
    set(fb_opt.h_kreuz(1), ...
        'xData',dscr_int(end)+fb_opt.cross_x(:,1), ...
        'yData',dtct_int(end)+fb_opt.cross_y(:,1));
    set(fb_opt.h_kreuz(2), ...
        'xData',dscr_int(end)+fb_opt.cross_x(:,2), ...
        'yData',dtct_int(end)+fb_opt.cross_y(:,2));
    drawnow;
  end
  
  if pp==mrk.pos(evt),
    if sign(dscr_out(ptr+fb_opt.toc_idx))==[-1 1]*mrk.y(:,evt),
      col= [0 .75 .1];
    else
      col= 'r';
    end
    set(fb_opt.h_toc, 'color',col);
    pause;
    set(fb_opt.h_toc, 'color','b');
    evt= evt+1;
  end
  
end
