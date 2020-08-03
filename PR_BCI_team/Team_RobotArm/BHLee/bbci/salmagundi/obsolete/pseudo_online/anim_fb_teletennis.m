function fb_opt= animate_fb_teletennis(fb_opt, ptr, dscr_out,varargin)
%fb_opt= animate_fb_teletennis(fb_opt, ptr, dscr_out)

%% share variables with feedback_run
global mrk pp first_test_event time_line trigger

%% static variables
persistent dscr_out_ma ev highlight_countdown oldPos rmax rmin
persistent ht_marker hp_left hp_right ht_title ha1 ha2

if isempty(trigger)
  trigger = cell(2,2);
  trigger{1,1} = 'L';
  trigger{1,2} = [1,65,70];
  trigger{2,1} = 'R';
  trigger{2,2} = [2,74,192];
end

% $$$ if isempty(time_line)
% $$$   time_line = zeros(1,size(dscr_out,2));
% $$$ end


colors = [1 0 0;0 1 0; 0 0 1];

racket_x_perc= 1/10;  %% half width of racket in percent of axis width
MARKER_VIEW= [-1 4];

if isequal(ptr, 'init'),
  fb_opt.iv= getIvalIndices(fb_opt.ival, fb_opt.fs);
%  fb_opt.toc_idx= round(fb_opt.toc*fb_opt.fs/1000);
  dscr_out_ma= zeros(size(dscr_out));
  ev= first_test_event;
  
  clf;
  ha1= axes('position', [0 .01 .999 .09]); box on;
  set(gca, 'xTick',0, 'xTickLabel','', 'xLim',1000*MARKER_VIEW, ...
           'xDir','reverse','yTick',[], 'yLim',[-1 1]);
  maxMarkers= 5*diff(MARKER_VIEW);        %% 5 markers per second is maximum
  ht_marker= text(NaN*ones(1,maxMarkers), zeros(1,maxMarkers), '');
  set(ht_marker, 'fontSize',28, 'fontWeight','bold');
  
  ha2= axes('position', [0 .11 1 .83]);
  set(gca, 'xLim',fb_opt.xLim, 'yLim',[0 1]);
  ht_title= title(' ');
  set(ht_title, 'fontSize',14);
  line([0 0]', [ylim]', 'color','k');
  cx= diff(xlim)*racket_x_perc;
  fb_opt.racket_x= [-cx cx]';
  fb_opt.racket_y= [0.1 0.1]';
  fb_opt.h_racket= ...
      line(fb_opt.racket_x, fb_opt.racket_y, ...
           'color','b', 'lineWidth',20);
  fb_opt.racket_x_rng= fb_opt.xLim + [cx -cx];
  axis off;
  oldPos = mean(xlim);
  xl = xlim;
  rmax = xl(2)-cx; rmin=xl(1)+cx;
else
  iv= ptr + fb_opt.iv;
  iv(find(iv<1))= 1;
  p0= max(1, ptr-fb_opt.integrate+1);
  dscr_out_ma(ptr)= mean(dscr_out(p0:ptr));
  xx= min(max(dscr_out_ma(ptr), fb_opt.racket_x_rng(1)), ...
          fb_opt.racket_x_rng(2));
  oldPos = fb_opt.relativ*oldPos + xx;
  if oldPos>rmax, oldPos = rmax;end
  if oldPos<rmin, oldPos = rmin;end
  set(fb_opt.h_racket, ...
<<<<<<< animate_fb_teletennis.m
      'xData',oldPos+fb_opt.racket_x(:,1));
  set(ht_title, 'string',sprintf('time: %.1f s', time_line(ptr)));
  
  
=======
      'xData',xx+fb_opt.racket_x(:,1));
%  set(ht_title, 'string',sprintf('time: %.1f s', time_line(ptr)));

>>>>>>> 1.3
  axes(ha1);
  mk= (pp-mrk.pos)/mrk.fs;
  inWindow= find(mk>MARKER_VIEW(1) & mk<MARKER_VIEW(2));
  nMark= length(inWindow);
  to = mrk.toe(inWindow);
  trig = cat(1,trigger{:,2});
  for ii= 1:nMark,
    [iii,jjj] = find(to(ii)==trig);
    if ~isempty(iii)
      x= (mrk.pos(inWindow(ii))-pp)*1000/mrk.fs;
      set(ht_marker(ii), 'position', [-x 0 0], 'string',trigger{iii,1}, ...
			'color',colors(iii,:));
    end
  end
  set(ht_marker(nMark+1:end), 'position',[NaN 0 0]);

  axis(ha2);
  drawnow;
end
