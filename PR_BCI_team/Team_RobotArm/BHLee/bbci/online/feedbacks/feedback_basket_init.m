function [hh,other_args] = feedback_basket_init(fig, opt);

fast_fig= {};
fast_axis= {'clipping','off', 'hitTest','off', 'interruptible','off', ...
            'drawMode','fast'};
fast_obj= {'eraseMode','xor', 'hitTest','off', 'interruptible','off'};

clf;
set(fig, 'Menubar','none', 'Resize','off', ...
         'position',opt.position);
set(fig,'color',opt.background_color, ...
        'renderer','painters', 'DoubleBuffer','on', ...
        'Pointer','custom', 'PointerShapeCData',ones(16)*NaN);

set(gca, fast_axis{:});
switch(opt.direction),
 case 'downward',
  if strcmp(opt.orientation, 'landscape'),
    set(gca, 'yDir','reverse');
    moveIdx= 2;
  else
    moveIdx= 1;
  end
 case 'right',
  if strcmp(opt.orientation, 'landscape'),
    moveIdx= 1;
  else
    moveIdx= 2;
  end
 case 'upward',
  if strcmp(opt.orientation, 'landscape'),
    moveIdx= 2;
  else
    set(gca, 'xDir','reverse');
    moveIdx= 1;
  end
 case 'left',
  if strcmp(opt.orientation, 'landscape'),
    set(gca, 'xDir','reverse');
    moveIdx= 1;
  else
    set(gca, 'yDir','reverse');
    moveIdx= 2;
  end
end
xLim= [-1 1];
yLim= [-1 1];
incr= 2/(1/opt.target_height-1);
if moveIdx==2,
  yLim(2)= yLim(2)+incr;
else
  xLim(2)= xLim(2)+incr;
end
set(gca, 'position',[0 0 1 1], 'XLim',xLim, 'YLim',yLim, ...
         'visible','off', 'nextPlot','add');

%% prepare cursor
h_cursor= plot(0, 0, ...
               'marker',opt.cursor_type, ...
               'markerSize',opt.cursor_size, fast_obj{:});
if strcmp(opt.cursor_type, 'o'),
  set(h_cursor, 'markerFaceColor',opt.cursor_color, ...
                'markerEdgeColor','none');
else
  set(h_cursor, 'markerFaceColor','none', ...
                'markerEdgeColor',opt.cursor_color);
end
  
%% prepare text objects (counter and messages)
h_text= text(0, 0, 'o', 'visible','off', 'fontUnits','normalized', ...
             'fontSize',opt.msg_fontsize,'FontName','helvetica');
h_counter= text(opt.counter_pos(1), opt.counter_pos(2), '0:0', ...
                'fontUnits','normalized', ...
                'fontSize',opt.counter_fontsize,'FontName','helvetica');
set(h_counter, 'visible',opt.show_counter);
set([h_counter, h_text], 'horizontalAli','center', 'verticalAli','middle');
if strcmp(opt.orientation, 'portrait'),
  set([h_counter, h_text], 'rotation',90);
end

%% prepare baskets
tw= 2/(opt.targets-2 + 2*opt.outerTarget_size);
sec= [-1, -1+tw*opt.outerTarget_size:tw:1-tw*opt.outerTarget_size, 1];
h_rect= zeros(opt.targets, 1);
h_rect2= zeros(opt.targets, 1);
basket= zeros(opt.targets, 4);
basket2= zeros(opt.targets, 4);
for rr= 1:opt.targets,
  if moveIdx==1,
    basket(rr,:)= [1, sec(rr)+(rr>1)*eps, 1+incr, sec(rr+1)];
    basket2(rr,:)= [1+incr*(1-opt.next_target_width), sec(rr)+(rr>1)*eps, 1+incr, sec(rr+1)];
  else
    basket(rr,:)= [sec(rr)+(rr>1)*eps, 1, sec(rr+1), 1+incr];
    basket2(rr,:)= [sec(rr)+(rr>1)*eps, 1+incr*(1-opt.next_target_width), sec(rr+1), 1+incr];
  end
  h_rect(rr)= patch(basket(rr,[1 3 3 1]), basket(rr,[2 2 4 4]), ...
                    opt.nontarget_color);
  h_rect2(rr)= patch(basket2(rr,[1 3 3 1]), basket2(rr,[2 2 4 4]), ...
                    opt.nontarget_color);
end
if opt.next_target==0
  set(h_rect2,'Visible','off');
end
set(h_rect, 'edgeColor','none');
set(h_rect2, 'edgeColor','none');
for rr= 2:opt.targets,
  if moveIdx==1,
      line([1 1+incr], sec([rr rr]), 'color','k', 'lineWidth',3);
  else
    line(sec([rr rr]), [1 1+incr], 'color','k', 'lineWidth',3);
  end
end

%% prepare fixation cross
pos= get(gcf, 'position');
xyr= pos(3)/pos(4);
h_cross= line([-1 1; 0 0]'*opt.fixation_size + opt.fixation_position(1), ...
              [0 0; -1 1]'*opt.fixation_size*xyr + opt.fixation_position(2));
set(h_cross, 'color',opt.fixation_color, ...
             'lineWidth',opt.fixation_linewidth, ...
               'visible',opt.show_fixation);
% here come the return values:
other_args = {basket,moveIdx};
hh = [fig,gca,h_cursor,h_cross',h_counter,h_text,h_rect',h_rect2'];


