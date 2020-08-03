function [hh,other_args] = feedback_speller_basket_init(fig, opt);

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
  

%% prepare baskets
% opt.targets only gives the maximum number of boxes; fewer can occur!
% h_rect contains rectangles for each number between 2 and opt.targets.
% likewise h_line and h_boxtext.
h_rect= zeros(opt.targets*(opt.targets+1)/2-1, 1);
h_line= zeros(opt.targets*(opt.targets-1)/2, 1);
h_boxtext = zeros(opt.targets*(opt.targets+1)/2-1, 1);
for i = 2:opt.targets
  basket{i-1}= zeros(i, 4);
  if i == 2
    tw = 1;
    sec = [-1,0,1];
    mid = [-.5,.5];
  else
    tw= (2 -2*opt.outerTarget_size)/(i-2 );
    sec= [-1, -1+opt.outerTarget_size:tw:1-opt.outerTarget_size, 1];
    mid = [-1+opt.outerTarget_size/2, ...
           -1+opt.outerTarget_size+tw/2:tw:1-opt.outerTarget_size-tw/2, ...
           1-opt.outerTarget_size/2];
  end
  for rr= 1:i,
    if moveIdx==1,
      basket{i-1}(rr,:)= [1, sec(rr)+(rr>1)*eps, 1+incr, sec(rr+1)];
      h_rect(rr + (i-1)*i/2-1)= patch(basket{i-1}(rr,[1 3 3 1]), ...
                                      basket{i-1}(rr,[2 2 4 4]), ...
                                      opt.nontarget_color);
      h_boxtext(rr + (i-1)*i/2-1) = text(1+incr/2, mid(rr),'', ...
                                         'fontUnits','normalized',...
                                         'fontSize', opt.boxtext_fontsize);
    else
      basket{i-1}(rr,:)= [sec(rr)+(rr>1)*eps, 1, sec(rr+1), 1+incr];
      h_rect(rr + (i-1)*i/2-1)= patch(basket{i-1}(rr,[1 3 3 1]), ...
                                      basket{i-1}(rr,[2 2 4 4]), ...
                                      opt.nontarget_color);
      h_boxtext(rr + (i-1)*i/2-1) = text(mid(rr), 1+incr/2, '', ...
                                         'fontUnits','normalized',...
                                         'fontSize', opt.boxtext_fontsize);
    end
  end
  for rr= 2:i,
    if moveIdx==1,
      h_line(rr-1+ (i-2)*(i-1)/2)=line([1 1+incr], sec([rr rr]), 'color','k', 'lineWidth',3);
    else
      h_line(rr-1+ (i-2)*(i-1)/2)=line(sec([rr rr]), [1 1+incr], 'color','k', 'lineWidth',3);
    end
  end

end
set(h_rect, 'edgeColor','none');
set(h_rect(1:(end-opt.targets)), 'visible', 'off');
set(h_line(1:(end-opt.targets+1)), 'visible', 'off');
%% prepare text objects (counter, messages, textbuffer and text in baskets.)
h_text= text(0, 0, 'o', 'visible','off', 'fontUnits','normalized', ...
             'fontSize',opt.msg_fontsize);
h_buffertext= text(opt.buffertext_pos(1), opt.buffertext_pos(2), '', ...
                'fontUnits','normalized', ...
                'fontSize',opt.buffertext_fontsize);
set([h_buffertext, h_text, h_boxtext'], 'horizontalAli','center', 'verticalAli','middle');
if strcmp(opt.orientation, 'portrait'),
  set([h_buffertext, h_text, h_boxtext'], 'rotation',90);
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
hh = [fig,gca,h_cursor,h_cross',h_buffertext,h_text,h_rect',h_boxtext',h_line'];


