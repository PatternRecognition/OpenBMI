function [H, hex]= feedback_hexawrite_init(fig, opt);

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

o.bottom_margin= 0.01;

hexheight= opt.hexradius/2/tan(2*pi/12);
hex.ursprung= [0; -1 + 3*hexheight + o.bottom_margin];
hexangles= 0:pi/3:2*pi;
for ai= 1:6,
  angie= hexangles(ai)+pi/6;
  cx(ai)= opt.hexradius*sin(angie);
  cy(ai)= opt.hexradius*cos(angie);
  angie= hexangles(ai);
  dx= 2*hexheight*sin(angie);
  dy= 2*hexheight*cos(angie);
  H.biglabel(ai)= text(hex.ursprung(1)+dx, hex.ursprung(2)+dy, ' ');
  H.hex(ai)= fb_hexawrite_drawHex(hex.ursprung+[dx;dy], angie-pi+2*pi/12, ...
				   opt, 'label',opt.labelset(ai,:));
  set([H.hex.label], fast_text{:});
end
set(H.biglabel, 'HorizontalAli', 'center', 'VerticalAli','middle', ...
                'FontUnits','normalized', ...
		opt.label_spec{:}, opt.biglabel_spec{:}, fast_text{:});

cx= cx + hex.ursprung(1);
cy= cy + hex.ursprung(2);
H.cake= line(cx([1 4; 2 5; 3 6]'), cy([1 4; 2 5; 3 6]'), opt.hex_spec{:});
if ~opt.show_cake,
  set(H.cake, 'Visible','off');
end

H.control_meter= patch(opt.control_x([1 2 2 1]), ...
                       -opt.control_h*[1 1 1 1], 'r');
set(H.control_meter, 'EdgeColor','none', fast_obj{3:end}, ...
                  opt.control_meter_spec{:});
framedist= 0.01*[-1 1];
xx= opt.control_x + framedist;
yy= [-1 1]*opt.control_h + framedist;
H.control_outline= line(xx([1 2 2 1; 2 2 1 1]'), ...
                        yy([1 1 2 2; 1 2 2 1]'));
yy= -opt.control_h + (opt.threshold_turn+1)*opt.control_h + framedist;
H.control_threshold_turn= line(xx, yy([1 1]));
yy= -opt.control_h + (opt.threshold_move+1)*opt.control_h +framedist;
H.control_threshold_move= line(xx, yy([1 1]));
set([H.control_threshold_turn H.control_threshold_move], ...
    'Color','k', 'LineWidth',2, fast_obj{3:end});
set([H.control_outline], 'Color','k', 'LineWidth',2, fast_obj{3:end});
if ~opt.show_control,
  set([H.control_meter H.control_threshold_turn H.control_threshold_move ...
       H.control_outline'], 'Visible','off');
end

H.msg= text(hex.ursprung(1), hex.ursprung(2), ' ');
set(H.msg, 'HorizontalAli','center', 'VerticalAli','middle', ...
		 'FontUnits','normalized', ...
		 opt.msg_spec{:}, fast_text{:});

arrow= fb_hexawrite_arrow(-pi/12, 0, opt);
H.arrow= patch(hex.ursprung(1)+arrow(1,:), hex.ursprung(2)+arrow(2,:), 'k');
set(H.arrow, opt.arrow_spec{:});
set([H.arrow], fast_obj{3:end});
set([H.hex.label, H.hex.outline], fast_obj{3:end});

textfield_height= 2 - 6*hexheight - o.bottom_margin - ...
    opt.textfield_to_hex_gap - opt.textfield_to_top_gap;
textfield_width= 6*hexheight * opt.textfield_width;
textfield_rect= [-textfield_width/2 ...
                 1-opt.textfield_to_top_gap-textfield_height ...
                 textfield_width/2 ...
                 1-opt.textfield_to_top_gap];

H.textfield_box= line(textfield_rect([1 3 3 1; 3 3 1 1]), ...
                      textfield_rect([2 2 4 4; 2 4 4 2]));
set(H.textfield_box, opt.textfield_box_spec{:});

ht= text(0, 0, {'MMMMMMM','MMMMMMM','MMMMMMM','MMMMMMM','MMMMMMM'});
set(ht, 'FontName','Courier New', 'FontUnits','normalized', ...
        'FontSize',0.06, 'FontWeight','bold', opt.text_spec{:});
rect= get(ht, 'Extent');
char_width= rect(3)/7;
linespacing= rect(4)/5;
char_height= linespacing*0.85;
hex.textfield_nLines= floor((textfield_height-2*char_height)/linespacing)+2;
hex.textfield_nChars= floor(textfield_width/char_width);
delete(ht);
H.textfield= text(0, mean(textfield_rect([2 4])), {' '});
set(H.textfield, 'HorizontalAli','center', 'VerticalAli','middle', ...
                 'FontName','Courier New', 'FontUnits','normalized', ...
                'FontSize',0.06, 'FontWeight','bold', ...
                 opt.text_spec{:}, fast_text{:});
