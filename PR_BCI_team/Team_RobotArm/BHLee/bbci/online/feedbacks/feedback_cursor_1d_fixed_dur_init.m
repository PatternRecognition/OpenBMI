function [H, cfd]= feedback_cursor_1d_fixed_dur_init(fig, opt);

%% most of those things do not really help
fast_fig= {'Clipping','off', 'HitTest','off', 'Interruptible','off'};
fast_axis= {'Clipping','off', 'HitTest','off', 'Interruptible','off', ...
            'DrawMode','fast'};
fast_obj= {'EraseMode','xor', 'HitTest','off', 'Interruptible','off'};
fast_text= {'HitTest','off', 'Interruptible','off', 'Clipping','off', ...
            'Interpreter','none'};

clf;
set(fig, 'Menubar','none', 'Renderer','painters', 'DoubleBuffer','on', ...
	 'Position',opt.position, ...
	 'Color',opt.background, ...
	 'Pointer','custom', 'PointerShapeCData',ones(16)*NaN, fast_fig{:});

d= 2*opt.target_dist;
w0= 2*opt.target_width;
w= w0/(1-w0);
nw= opt.next_target_width*w;
target_rect= [-1-w  -1+d -1 1-d;
              1 -1+d 1+w 1-d];
target_rect2= [-1-nw   -1+d -1 1-d;
               1 -1+d 1+nw 1-d];
set(gca, 'Position',[0 0 1 1], ...
         'XLim',[-1-w 1+w]*1.01, 'YLim',[-1 1], ...
         'Visible','off', fast_axis{:});
%axis off;
cfd.target_width= w;

for tt= 1:2,
  H.target(tt)= patch(target_rect(tt,[1 3 3 1]), ...
                      target_rect(tt,[2 2 4 4]), ...
                      opt.color_nontarget);
  H.next_target(tt)= patch(target_rect2(tt,[1 3 3 1]), ...
                           target_rect2(tt,[2 2 4 4]), ...
                           opt.color_nontarget);
end
H.punchline= line([-1 1; -1 1], [-1+d -1+d; 1-d 1-d], ...
                  'Color',[0 0 0], ...
                  'LineWidth', 2, ...
                  opt.punchline_spec{:});
if ~opt.punchline,
  set(H.punchline, 'Visible','off');
end
H.center= patch([-1 1 1 -1]*opt.center_size, ...
                [-1 -1 1 1]*opt.center_size, opt.color_center);

set([H.target H.next_target H.center], 'EdgeColor','none', fast_obj{3:end});
H.fixation = line(0, 0, 'Color','k', 'LineStyle','none');
set(H.fixation, fast_obj{3:end}, opt.fixation_spec{:}, 'Visible','off');

if opt.rate_control & ~strcmpi(opt.timeout_policy,'hitiflateral'),
  set(H.center, 'Visible','off');
end

H.msg= text(0, 0, ' ');
H.msg_punch= text(0, -0.5, ' ');
set([H.msg H.msg_punch], ...
     'HorizontalAli','center', 'VerticalAli','middle', ...
     'FontUnits','normalized', opt.msg_spec{:}, fast_text{:});
set(H.msg_punch, opt.points_spec{:});

H.points= text(-0.5, 0.99, 'HIT: 0');
H.points(2)= text(0.5, 0.99, 'MISS: 0');
set(H.points, 'VerticalAli','top', 'HorizontalAli','center', ...
          'FontUnits','normalized', opt.points_spec{:}, fast_text{:});

H.cursor = line(0, 0, 'Color','k', 'LineStyle','none');
set(H.cursor, fast_obj{3:end}, opt.cursor_inactive_spec{:});

if opt.next_target==0
  set(H.next_target, 'Visible','off');
end
