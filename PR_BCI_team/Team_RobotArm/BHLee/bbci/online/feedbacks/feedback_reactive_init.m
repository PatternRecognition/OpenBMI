function handle = feedback_reactive_init(fig,fb_opt);

clf;
set(fig, 'Menubar','none', 'Resize','off', ...
         'position',fb_opt.position);
hold on;
set(fig,'color',[1 1 1], 'DoubleBuffer','on', ...
        'Pointer','custom', 'PointerShapeCData',ones(16)*NaN);
set(get(fig,'Children'), 'position',[0 0 1 1], ...
                  'XLim',[-1 1], 'YLim',[-1 1]);
axis off



mess = text(0,0,'');
set(mess,'HorizontalAlignment','center','VerticalAlignment','middle');
set(mess,'FontUnits','normalized','FontSize',fb_opt.font_size);
set(mess,'Color',fb_opt.font_color);
set(mess,'Visible','off');

classes = reactive_task(fb_opt,'init');

fix_point = patch([-1 1 1 -1 -1]*fb_opt.fixPointWidth,[-1 -1 1 1 -1]*fb_opt.fixPointWidth,fb_opt.fixPointColor);

set(fix_point,'LineWidth',0.001,fb_opt.fixPoint_params{:});
if fb_opt.fixPoint
  set(fix_point,'Visible','on');
else
  set(fix_point,'Visible','off');
end

score = text(0,0.9,'');
set(score,'HorizontalAlignment','center','VerticalAlignment','middle');
set(score,'FontUnits','normalized','FontSize',fb_opt.score_size);
set(score,'Color',fb_opt.score_color);
set(score,'Visible','off');


handle = [mess,score,fix_point,classes];








