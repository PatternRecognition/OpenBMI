function fb_opt= animate_fb_cross2d(fb_opt, ptr, dscr_out, dtct_out, comb_out)
%fb_opt= animate_fb_cross2d(fb_opt, ptr, dscr_out, dtct_out, comb_out)

%% share variables with feedback_run
global mrk pp

%% static variables
persistent dtct_out_ma dscr_out_ma ev highlight_countdown 
persistent ht_marker hp_left hp_right ht_title ha1 ha2

cross_x_perc= 1/20;  %% half width of cross in percent of axis width
HIGHLIGHT_TIME= 300;
MARKER_VIEW= 3;

if isequal(ptr, 'init'),
  highlight_countdown= 0;
  fb_opt.iv= getIvalIndices(fb_opt.ival, fb_opt.fs);
  dtct_out_ma= zeros(size(dscr_out));
  dscr_out_ma= zeros(size(dscr_out));
  ev= first_test_event;
  
  clf;
  ha1= axes('position', [0 .01 .999 .09]); box on;
  set(gca, 'xTick',0, 'xTickLabel','', 'xLim',[-1000 1000]*MARKER_VIEW, ...
           'yTick',[], 'yLim',[-1 1]);
  maxMarkers= 10*MARKER_VIEW;           %% 5 markers per second is maximum
  ht_marker= text(NaN*ones(1,maxMarkers), zeros(1,maxMarkers), '');
  set(ht_marker, 'fontSize',28, 'fontWeight','bold');
  
  ha2= axes('position', [0 .11 1 .83]);
  set(gca, 'xLim',fb_opt.xLim, ...
           'yLim',fb_opt.yLim);
  ma= fb_opt.xLim(2);
  xPat= [0.25 1 ma ma .25 .25];
  yPat= [.75 0 0 ma ma .75];
  ht_title= title(' ');
  set(ht_title, 'fontSize',14);
  hp_left= patch(-xPat, yPat, [1 .8 .8]*0.9);
  hp_right= patch(xPat, yPat, [.8 1 .8]*0.9);
  set(hp_left, 'edgeColor','none');
  set(hp_right, 'edgeColor','none');
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
  fb_opt.cross_x= 1.5*[-cx cx; 0 0]';
  fb_opt.cross_y= 1.5*[0 0; -cy cy]';
  fb_opt.h_kreuz= ...
      line(fb_opt.cross_x, fb_opt.cross_y, ...
           'color','b', 'lineWidth',3);
  axis off;
  
else
  iv= ptr + fb_opt.iv;
  iv(find(iv<1))= 1;
  p0= max(1, ptr-fb_opt.integrate+1);
  dscr_out_ma(ptr)= mean(dscr_out(p0:ptr));
  dtct_out_ma(ptr)= mean(dtct_out(p0:ptr));
  set(fb_opt.h_schweif, 'xData',dscr_out_ma(iv), 'yData',dtct_out_ma(iv));
  set(fb_opt.h_kreuz(1), ...
      'xData',dscr_out_ma(ptr)+fb_opt.cross_x(:,1), ...
      'yData',dtct_out_ma(ptr)+fb_opt.cross_y(:,1));
  set(fb_opt.h_kreuz(2), ...
      'xData',dscr_out_ma(ptr)+fb_opt.cross_x(:,2), ...
      'yData',dtct_out_ma(ptr)+fb_opt.cross_y(:,2));
  if ~isempty(time_line),
    set(ht_title, 'string',sprintf('time: %.1f s', time_line(ptr)));
  end
  
  if comb_out(ptr)~=0,
    if highlight_countdown>0,       %% switch off ongoing highlightning
      set(hp_left, 'faceColor',[1 0.8 0.8]*0.9);
      set(hp_right, 'faceColor',[0.8 1 0.8]*0.9);
    end
    highlight_countdown= HIGHLIGHT_TIME;
%    highlight_class= comb_out(ptr);
    if comb_out(ptr)==-1,
      set(hp_left, 'faceColor',[1 0.5 0.5]);
    elseif comb_out(ptr)==1
      set(hp_right, 'faceColor',[0.5 .9 0.5]);
    end
  end
  
  if highlight_countdown>0,
    highlight_countdown= highlight_countdown - 1000/fb_opt.fs;
    if highlight_countdown<=0,
      set(hp_left, 'faceColor',[1 0.8 0.8]*0.9);
      set(hp_right, 'faceColor',[0.8 1 0.8]*0.9);
    end
  end

  if isfield(mrk,'y'),
   axes(ha1);
   inWindow= find(abs(mrk.pos-pp)/mrk.fs<MARKER_VIEW);
   nMark= length(inWindow);
   for ii= 1:nMark,
     x= (mrk.pos(inWindow(ii))-pp)*1000/mrk.fs;
%     set(ht_marker(ii), 'position',[x 0 0], ...
%                        'string',char('LR'*mrk.y(:,inWindow(ii))));
     if mrk.y(1,inWindow(ii)),
       set(ht_marker(ii), 'position', [x 0 0], 'string','L', ...
                         'color',[1 0 0]);
     else
       set(ht_marker(ii), 'position', [x 0 0], 'string','R', ...
                         'color',[0 .7 0]);
     end
   end
   set(ht_marker(nMark+1:end), 'position',[NaN 0 0]);
   axis(ha2);
  end
  drawnow;
end
