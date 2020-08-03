function out= stimutil_reactometer(msec, varargin)
%STIMUTIL_REACTOMETER - Show Reaction Time
%
%Synopsis:
% HANDLES= stimutil_reactometer('init', <OPT>)
% stimutil_reactometer('close')
% stimutil_reactometer(MSEC)
%
%Arguments:
% MSEC: Reaction time in msec. Use negative values for false reactions.
% OPT: struct or property/value list of optional arguments:
%   _TO_BE_SPECIFIED_
%
%Returns:
% HANDLES: Handles to graphical objects. Can be used to switch propery
%   'Visible', 'off'.

% blanker@cs.tu-berlin.de, Jul-2007


persistent memo

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'msec_range', [200 700], ...
                  'tick_spec', {'FontSize',0.04}, ...
                  'color_hit', [0 0.8 0], ...
                  'color_miss', [0.6 0 0]);
              
% some slide size and location params
width = 0.18;
height = 0.45;
bl = [0.0, 0.22];  % buttom point of the slide

if ischar(msec),
  switch(lower(msec)),
   case 'init',
    memo_ax= gca;
    memo= [];
    memo.nSteps= 101;
    ticks= [opt.msec_range(1):100:opt.msec_range(2)];
    iTicks= round((ticks-opt.msec_range(1))/diff(opt.msec_range)*(memo.nSteps-1));
   
    memo.h_background= axes('position',[0 0 1 1]);
    set(memo.h_background, 'Visible','off', ...
                      'XLim',[-1 1], ...
                      'YLim',[-0.1 1]);

%     ### keep this - to make a color graded patch
%     bx = [bl(1)-width/2; bl(1)+width/2];  % x coords
%     by = [linspace(bl(2) ,bl(2) + height, memo.nSteps)];  % y coords
%     by_sub_len = by(2) - by(1);
%     cval= linspace(0, 1, memo.nSteps);
%     memo.h_patch1= patch(repmat([bx; flipud(bx)],1, memo.nSteps), ...
%                         [repmat(by,2,1); repmat(by+by_sub_len, 2,1)], ...
%                         [cval]);
%     set(memo.h_patch1, 'EdgeColor','none');
    
    cx1 = bl(1) - width/2 - 0.05;
    cx2 =  bl(1) + width/2 + 0.05;
    cy = bl(2) + height .* iTicks./iTicks(end);
    tick_width = 0.02;
    tick_color = [0.4 0.4 0.4];
    
 
    colormap(cmap_hsv_fade(2,[1/6 -1/6], 1, 0.8));

    
    memo.h_patch = line([cx1, cx2; cx1, cx2], [bl(2), bl(2); bl(2)+height,bl(2)+height ], 'Color',tick_color,'LineWidth',3);
   
    tick_spec= {'HorizontalAli','center', ...
                'VerticalAli','baseline', ...
                'FontUnit','normalized', ...
                'Color',tick_color, ...
                opt.tick_spec{:}};
    memo.h_text= text([(cx1-0.06)*ones(1,length(iTicks)),(cx2+0.06)*ones(1,length(iTicks))], ...
                      [cy, cy], ...
                      repmat(cellstr(int2str(ticks')), 1,2), tick_spec{:});
                  
    memo.h_ticks= line([repmat([cx1-tick_width/2; cx1+tick_width],1,length(iTicks)), ...
                          repmat([cx2-tick_width/2; cx2+tick_width],1,length(iTicks))] - 0.01, ...
                       repmat(cy,2,2), ...
                       'Color',tick_color, 'LineWidth', 1);
    
    axis equal;
    set(gca, 'XLimMode','manual', 'YLimMode','manual');
    memo.alpha= memo;
    memo.opt= opt;  
    msec= mean(opt.msec_range);
  
   case 'close',
    delete(memo.h_background); 
    return;
    
   otherwise,
    error('Wrong syntax');
  end
end

if abs(msec)<memo.opt.msec_range(1),
  msec= sign(msec) * (memo.opt.msec_range(1) - 0.05*diff(memo.opt.msec_range));
end
if abs(msec)>memo.opt.msec_range(2),
  msec= sign(msec) * (memo.opt.msec_range(2) + 0.05*diff(memo.opt.msec_range));
end

if ~isfield(memo, 'h_slide'),  %% 'init' case
  memo.h_slide= patch([bl(1)-width/2; bl(1)+width/2; bl(1)+width/2; bl(1)-width/2],...
                      [bl(2); bl(2); bl(2)+height; bl(2)+height], 'k');
  set(memo.h_slide, 'EdgeColor','none');
  out= [memo.h_slide; memo.h_text; memo.h_patch; memo.h_ticks];
  set(out, 'Visible','off');
  %set(memo.h_slide, 'Visible','off');
  axis(memo_ax);
else
  set([memo.h_slide; memo.h_text; memo.h_patch; memo.h_ticks], ...
      'Visible','on');
end

if msec<0,
  col= memo.opt.color_miss;
else
  col= memo.opt.color_hit;
end
%if abs(msec)>=mean(memo.opt.msec_range),
%  col= col*0.6;
%end
msec_height = (abs(msec) - memo.opt.msec_range(1)) / ...
               (memo.opt.msec_range(end) - memo.opt.msec_range(1)) * (height);
set(memo.h_slide, 'XData',[bl(1)-width/2; bl(1)+width/2; bl(1)+width/2; bl(1)-width/2],...
                  'YData',  [bl(2); bl(2); bl(2)+msec_height; bl(2)+msec_height], ...
                  'FaceColor',col);
drawnow;
