function [H, dash]= feedback_dashawrite_init(fig, opt);

%% most of those things do not really help
fast_fig= {'Clipping','off', 'HitTest','off', 'Interruptible','off'};
fast_axis= {'Clipping','off', 'HitTest','off', 'Interruptible','off', ...
            'DrawMode','fast'};
fast_obj= {'EraseMode','xor', 'HitTest','off', 'Interruptible','off'};
fast_text= {'HitTest','off', 'Interruptible','off', 'Clipping','off', ...
            'Interpreter','none'};

clf;
set(fig, 'Menubar','none', 'Interruptible','off', ...
	 'Renderer','painters', 'DoubleBuffer','on', ...
	 'Position',opt.position, ...
	 'Color',opt.background, ...
	 'Pointer','custom', 'PointerShapeCData',ones(16)*NaN, fast_fig{:});
set(gca, 'Position',[0 0 1 1], ...
         'XLim',[-1 1], 'YLim',[-1 1], fast_axis{:});
axis equal off;
XLim= get(gca, 'XLim');
set(gca, 'XLim', [-1 1]*diff(XLim)/2);
drawnow;

H.msg_ax= axes('position', [0 0 1 1]);
set(H.msg_ax, 'Visible','off', fast_axis{:});
H.msg= text(0.5, 0.5, ' ');
set(H.msg, 'HorizontalAli','center', 'VerticalAli','middle', ...
		 'Visible','on', 'FontUnits','normalized', ...
		 opt.msg_spec{:}, fast_text{:});

o.margin_top= 0.01;
o.margin_text_to_dash= 0.02;
o.margin_dash= 0.01;

H.textfield_ax= axes('position', ...
                     [(1-opt.textfield_width)/2, ...
                      1 - opt.textfield_height + o.margin_text_to_dash, ...
                      opt.textfield_width, ...
                      opt.textfield_height - o.margin_top ...
                        - o.margin_text_to_dash]);
set(H.textfield_ax, 'Visible','off', fast_axis{:});
textfield_rect= [0 0 1 1];
H.textfield_box= line(textfield_rect([1 3 3 1; 3 3 1 1]), ...
                      textfield_rect([2 2 4 4; 2 4 4 2]));
set(H.textfield_box, opt.textfield_box_spec{:});

ht= text(0.5, 0.5, {'MMMMM','M','M','M','M'});
set(ht, 'FontName','Courier New', 'FontUnits','normalized', opt.text_spec{:});
rect= get(ht, 'Extent');
char_width= rect(3)/5;
char_height= rect(4)/5;
delete(ht);
dash.textfield_nLines= floor(1/char_height);
dash.textfield_nChars= floor(1/char_width);
H.textfield= text(0.5, 0.5, {' '});
set(H.textfield, 'HorizontalAli','center', 'VerticalAli','middle', ...
                 'FontName','Courier New', ...
                 'FontUnits','normalized', ...
                 opt.text_spec{:}, fast_text{:});

H.pointerline_ax= axes('position', ...
                [o.margin_dash, ...
                 o.margin_dash, ...
                 1 - 2*o.margin_dash, ...
                 1 - o.margin_dash - opt.textfield_height]);
set(H.pointerline_ax, 'Visible','off');
H.pointerline= line([0 1], [0.5 0.5], 'Color','k', 'LineWidth',3, ...
                    opt.pointerline_spec{:});

H.pointer_ax= axes('position', ...
                   [o.margin_dash, ...
                    o.margin_dash, ...
                    opt.pointer_width, ...
                    1 - o.margin_dash - opt.textfield_height]);
set(H.pointer_ax, 'Visible','off');
H.pointer= patch([1 0 0], [0.5 0.1 0.9], [0 0.7 0]);
axis equal square
yl= get(H.pointer_ax, 'YLim');
yl= 0.5 + [-1 1]*diff(yl);
set(H.pointer_ax, 'XLim',[0 1], 'YLim',yl);

H.dash_ax= axes('position', ...
                [opt.pointer_width + o.margin_dash, ...
                 o.margin_dash, ...
                 1 - 2*o.margin_dash - opt.pointer_width, ...
                 1 - o.margin_dash - opt.textfield_height]);
set(H.dash_ax, 'XLim',[0 1+opt.fieldwidth], 'Visible','off');

%ext_list= 
k= 0;
h= text(0.5, 0.5, 'X');
set(h, 'FontUnit','Normalized');
for fs= 0.01:0.01:0.3,
  k= k+1;
  set(h, 'FontSize',fs);
%  drawnow;
  ext= get(h, 'Extent');
  ext_list(k, 1:2)= [fs, ext(4)];
end
dash.extent_factor= mean(ext_list(:,1)./ext_list(:,2));
delete(h);

y0= 1;
dash.sect(1)= y0;
dash.fieldheight= 1/length(opt.charset)*ones(1,length(opt.charset));
for ii= 1:length(opt.charset),
  y1= y0 - dash.fieldheight(ii);
  dash.sect(ii+1)= y1;
  H.field(ii)= patch([1 1+opt.fieldwidth 1+opt.fieldwidth 1], ...
                     [y1 y1 y0 y0], ...
                     opt.fieldcolor(mod(ii,2)+1,:));
  dash.lettersize(ii)= dash.fieldheight(ii) * dash.extent_factor;
  H.letter(ii)= text(1+opt.fieldwidth/2, (y0+y1)/2, opt.charset(ii));
  set(H.letter(ii), 'FontUnits','normalized', 'FontSize',dash.lettersize(ii));
  y0= y1;
end
%dash.sect(1)= 1+eps;
dash.sect(end)= 0;
set(H.letter, 'HorizontalAli','center', 'VerticalAli','middle', ...
              fast_text{:});

