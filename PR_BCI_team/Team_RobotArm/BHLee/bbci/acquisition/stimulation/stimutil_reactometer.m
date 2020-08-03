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
                  'msec_range', [300 700], ...
                  'alpha', 30, ...
                  'radius1', 0.8, ...
                  'radius2', 0.85, ...
                  'radius_ticks', 0.88, ...
                  'tick_spec', {'FontSize',0.07}, ...
                  'color_hit', [0 0.8 0], ...
                  'color_miss', [1 0 0], ...
                  'arrow_width', 0.075, ...
                  'arrow_backlength', 0.1, ...
                  'arrow_minlength', 0.20, ...
                  'arrow_headlength', 0.15, ...
                  'arrow_headwidth', 0.175);

if ischar(msec),
  switch(lower(msec)),
   case 'init',
    memo_ax= gca;
    memo= [];
    memo.nSteps= 101;
    memo.alpha_range= [pi-opt.alpha*pi/180, opt.alpha/180*pi];
    alpha= linspace(memo.alpha_range(1), memo.alpha_range(2), memo.nSteps);
    ticks= [opt.msec_range(1):100:opt.msec_range(2)];
    iTicks= round((ticks-opt.msec_range(1))/diff(opt.msec_range)*(memo.nSteps-1)+1);
  
    memo.h_background= axes('position',[0 0 1 1]);
    set(memo.h_background, 'Visible','off', ...
                      'XLim',[-1 1], ...
                      'YLim',[-0.1 1]);
    ax= cos(alpha);
    ay= sin(alpha);
    cval= linspace(0, 1, memo.nSteps);
    colormap(cmap_hsv_fade(2,[1/6 -1/6], 1, 0.8));
    memo.h_patch= patch([ax*opt.radius1, fliplr(ax)*opt.radius2], ...
                        [ay*opt.radius1, fliplr(ay)*opt.radius2], ...
                        [cval, fliplr(cval)]);
    set(memo.h_patch, 'EdgeColor','none');
    tick_spec= {'HorizontalAli','center', ...
                'VerticalAli','baseline', ...
                'FontUnit','normalized', ...
                opt.tick_spec{:}};
    memo.h_text= text(ax(iTicks)*opt.radius_ticks, ...
                      ay(iTicks)*opt.radius_ticks, ...
                      cellstr(int2str(ticks')), tick_spec{:});
    memo.h_ticks= line([ax(iTicks)*opt.radius1; ax(iTicks)*opt.radius2], ...
                       [ay(iTicks)*opt.radius1; ay(iTicks)*opt.radius2], ...
                       'Color','k');
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

arrow_length= 0.97;
angle= (abs(msec)-memo.opt.msec_range(1))/diff(memo.opt.msec_range) * ...
       diff(memo.alpha_range) + memo.alpha_range(1);
w= [cos(angle); sin(angle)]*memo.opt.radius1;
wn= [-w(2); w(1)]/sqrt(w'*w)*memo.opt.radius1;
bp= -w*memo.opt.arrow_backlength;
pp= w*arrow_length;
fp= w*(arrow_length-memo.opt.arrow_headlength);

arrow= [bp, -wn*memo.opt.arrow_width, ...
        pp, ...
        wn*memo.opt.arrow_width];


if ~isfield(memo, 'h_arrow'),  %% 'init' case
  memo.h_arrow= patch(arrow(1,:), arrow(2,:), 'k');
  set(memo.h_arrow, 'EdgeColor','none');
  out= [memo.h_arrow; memo.h_text; memo.h_patch; memo.h_ticks];
  set(out, 'Visible','off');
  axis(memo_ax);
else
  set([memo.h_arrow; memo.h_text; memo.h_patch; memo.h_ticks], ...
      'Visible','on');
end

if msec<0,
  col= memo.opt.color_miss;
else
  col= memo.opt.color_hit;
end
if abs(msec)>=mean(memo.opt.msec_range),
  col= col*0.6;
end

set(memo.h_arrow, 'XData',arrow(1,:)', ...
                  'YData',arrow(2,:)', ...
                  'FaceColor',col);
drawnow;
