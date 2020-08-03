function [hh,other_args] = feedback_speller_2d_init(fig, opt);

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

gray = [1,1,1]*0.8;
GRAY = [1,1,1]*0.8;
blue = [0 0 1];
green = [0 1 0];
red = [1 0 0];

hold on;
set(get(fig,'Children'), 'position',[0 0 1 1], ...
                  'XLim',[-1 1], 'YLim',[-1 1]);
axis off
xLim= [-1 1];
yLim= [-1 1];
moveIdx = 1+strcmp(opt.orientation,'landscape');
%incr= 2/(1/opt.target_height-1);
set(gca, 'position',[0 0 1 1], 'XLim',xLim, 'YLim',yLim, ...
         'visible','off', 'nextPlot','add');

%% background for free region center
h_pat = patch([-1 1 1 -1]*opt.free_region, ...
            [-1 -1 1 1]*opt.free_region, GRAY);
set(h_pat, 'EdgeColor','none');

if opt.free_region>=1,
  set(h_pat, 'Visible','off');
else
  set(h_pat, 'Visible','on');
end

%% prepare cursor
h_cursor= plot(0, 0,...
    'marker',opt.marker_nonactive(2), ...
    'Color',opt.marker_nonactive(1),...
    'markerSize', opt.marker_nonactive_size(1),...
    'LineWidth', opt.marker_nonactive_size(2), ...
    fast_obj{:});





%% parts from fb_2d:
d= opt.target_dist;
w= opt.target_width;
switch(opt.target_mode),
 case 1,
  target_rect= [0   d w 1-2*d;
                1-w d w 1-2*d];
      
 case 2,
  target_rect= [d 0   1-2*d w;
                d 1-w 1-2*d w];
 otherwise,
     error('opt.targetmode>2 not implemented yet.');
end
nTargets= size(target_rect,1);
h_reg= zeros(1, nTargets);
%% scale [0 1]-rects to [-1 1]-rects
target_rect(:,[3 4])= 2*target_rect(:,[3 4]);
target_rect(:,[1 2])= 2*target_rect(:,[1 2])-ones(nTargets,2);
target_rect= target_rect(:,[1 2 1 2]) + ...
    [zeros(nTargets,2) target_rect(:, [3 4])];

for tt= 1:nTargets,
  h_reg(tt)= patch(target_rect(tt,[1 3 3 1]), target_rect(tt,[2 2 4 4]), ...
                  gray);
end
set(h_reg, 'EdgeColor','none');

%% prepare speller-textfields
h_boxtext = zeros(2, 1);
    mid = opt.boxtext_mid;%[-.9,.9];
    height = [.15,0,-.15];
  for rr= 1:2,
    if moveIdx==1,
      h_boxtext(rr) = text(0, mid(rr),'', ...
                                         'fontUnits','normalized',...
                                         'fontSize', opt.boxtext_fontsize);
    else
		for j = 1:3
               h_boxtext((rr-1)*3+j) = text(mid(rr), height(j), '', ...
                                                 'fontUnits','normalized',...
                                                 'fontSize', opt.boxtext_fontsize);
		end    
    end
  end
  
%% prepare text objects (counter, messages, textbuffer.)
h_text= text(0, 0, 'o', 'visible','off', 'fontUnits','normalized', ...
             'fontSize',opt.msg_fontsize);
h_buffertext= text(opt.buffertext_pos(1), opt.buffertext_pos(2), ...
    untex(opt.buffertext(max(1,length(opt.buffertext)-opt.buffertext_len+1):end)), ...
                'fontUnits','normalized', ...
                'fontSize',opt.buffertext_fontsize);
set([h_buffertext, h_text, h_boxtext'], 'horizontalAli','center', 'verticalAli','middle');
if strcmp(opt.orientation, 'portrait'),
  set([h_buffertext, h_text, h_boxtext'], 'rotation',90);
end

%% counter
h_cou = text(0, 0, '');

set(h_cou,'HorizontalAlignment','center', 'FontSize',100,'FontName','utopia');


               
% here come the return values:
other_args = {target_rect};
hh = [fig,gca,h_cursor,h_cou,h_pat,h_buffertext,h_text,h_boxtext',h_reg];
