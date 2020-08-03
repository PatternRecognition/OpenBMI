function fb_opt= animate_fb_traces(fb_opt, ptr, dscr_out, dtct_out, comb_out)
%fb_opt= animate_fb_traces(fb_opt, ptr, dscr_out, dtct_out, comb_out)

%% share variables with feedback_run
global mrk pp first_test_event time_line goal

%% static variables
persistent Tov Tzo xx ev
persistent hpZoom hZoomCursor hOv hMark
persistent usedMarks errColor
persistent dtct_out_ma dscr_out_ma cfy_out


if isequal(ptr, 'init'),
  clf;
  yLimCfy= [-1.3 1.3];
  cfyColor= [1 0.75 0; 1 0 1];
  cfyColorLight= [1 0.8 0.5; 1 0.5 1];
  combiColor= hsv2rgb([0.5 1 0.7]);
  shdColor= [1 0.9 0.9; 0.8 1 0.8];
  errColor= [1 0 0; 1 0 0; 0 0.7 0.9];
  margin= {[0.06 0.08 0.06], [0.05 0 0.025]};
  
  Tzo= ceil(fb_opt.zoomSec*fb_opt.fs);
  xx= linspace(0, fb_opt.zoomSec, Tzo); %xx(end)= [];
  cfy_out= repmat(NaN, 2, Tzo);
  ev= first_test_event;
  
  if fb_opt.overview,
    ha= axes('position', [.04 .7 .93 .24]);
    Tov= ceil(fb_opt.overviewSec*fb_opt.fs);
    hOv= plot(linspace(0, fb_opt.overviewSec, Tov), repmat(NaN, 7, Tov));
    set(hOv(1), 'color', cfyColorLight(1,:));
    set(hOv(2), 'color', cfyColorLight(2,:));
    set(hOv(3), 'color', combiColor, 'lineWidth',2);
    set(hOv(4), 'color','k', 'lineWidth',2);
    set(hOv(5), 'lineStyle','none', 'marker','o');
    set(hOv(5), 'color',errColor(1,:), 'markerSize',6, 'lineWidth',2);
    set(hOv(6:7), 'lineStyle','none', 'marker','.');
    set(hOv(6), 'color',errColor(2,:), 'markerSize',15);
    set(hOv(7), 'color',errColor(3,:), 'markerSize',15);
    set(gca, 'xLim',[0 fb_opt.overviewSec], 'yLim',yLimCfy);
    ht= title(['classification overview  ' ...
               '[pos: right (green), neg: left (red)]']);
    set([ha ht], 'fontUnits','normalized');
    
    ha= axes('position', [.04 .06 .93 .52]);
  else
    ha= axes('position', [.04 .1 .93 .8]);
  end

  hold on;
  ht= title(['classifier output  [orange: +movement/-none, ' ...
             'magenta: +right/-left]']);
  set([ha ht], 'fontUnits','normalized');
  
  nMarks= 6;
  usedMarks= zeros(nMarks, 1);
  for im= 1:nMarks,
    hMark(im)= plot(NaN, yLimCfy(1), '.');
  end
  set(hMark, 'markerSize',15, 'markerEdgeColor','k', ...
             'markerFaceColor','b');

  h= line(xx([1 end]), [0 0]);
  set(h, 'color', 'k');
  h= line(repmat(xx([1 end]),2,1)', [.5 .5; -.5 -.5]');
  set(h, 'color', 'k', 'lineStyle', ':');
  hpZoom= plot(xx, cfy_out);
  for ip= 1:size(cfyColor,1),
    set(hpZoom(ip), 'color', cfyColor(ip,:));
  end
  set(gca, 'xLim',[0 fb_opt.zoomSec], 'yLim',yLimCfy);
  hZoomCursor= line([0 0], yLimCfy);
  set(hZoomCursor, 'color','k', 'lineWidth',5);
  hold off;
  box on;
  
else
  p0= max(1, ptr-fb_opt.integrate+1);
  dtct_out_ma(ptr)= mean(dtct_out(p0:ptr));
  dscr_out_ma(ptr)= mean(dscr_out(p0:ptr));
  ptr_mod= mod(ptr-1, Tzo)+1;
  vanish= 6;
  va= min(vanish, Tzo-ptr_mod);
  cfy_out(:,ptr_mod:ptr_mod+va)= ...
      [[dtct_out_ma(ptr); dscr_out_ma(ptr)], NaN*ones(2,va)];

  if mod(ptr-1, fb_opt.disp_step)~=0, return; end
  
  for ip= 1:length(hpZoom),
    set(hpZoom(ip), 'yData', cfy_out(ip,:));
  end
  set(hZoomCursor, 'xData', xx(mod(ptr, Tzo)+1)*[1 1]);

  if fb_opt.overview,
    iv= ptr-Tov+1:ptr;
    iv(find(iv<1))= 1;
    set(hOv(1), 'yData', dtct_out_ma(iv));
    set(hOv(2), 'yData', dscr_out_ma(iv));
    set(hOv(3), 'yData', 1.1*comb_out(iv));
    set(hOv(4), 'yData', goal(iv));
%   set(hOv(5), 'yData', errVec(1,iv));
%   set(hOv(6), 'yData', errVec(2,iv));
%   set(hOv(7), 'yData', errVec(3,iv));
  end
  
  if ~isempty(goal),
    iv_novel= ptr-fb_opt.disp_step+1 : ptr;
    iv_novel_mod= mod(iv_novel-1, Tzo)+1;
    iv_novel(find(iv_novel<1))= 1;
    
    takeAway= find(ismember(usedMarks, iv_novel_mod));
    for im= takeAway(:)',
      set(hMark(im), 'xData', NaN);
      usedMarks(im)= 0;
    end

    goalNovel= goal(iv_novel);
    xEvent= find([1; diff(goalNovel)]~=0 & abs(goalNovel)==1);
    for xe= xEvent(:)',
      im= min(find(~usedMarks));
      if isempty(im), warning('not enough shades'); break; end
      usedMarks(im)= ptr_mod;
      cl= goalNovel(xe)/2+1.5;
      marker_type= '<>';
      set(hMark(im), 'xData',xx(ptr_mod), 'marker',marker_type(cl));
    end
  end

  drawnow;
  if ev<length(mrk.pos) & ...
        pp >= mrk.pos(ev) + round(fb_opt.stop_delay/1000*mrk.fs),
    pause;
    ev= ev+1;
  end
end
