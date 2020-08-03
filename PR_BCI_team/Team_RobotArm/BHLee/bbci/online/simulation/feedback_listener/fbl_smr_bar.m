function opt = fbl_smr_bar(fig, opt, ctrl_fast, ctrl_slow);
%FEEDBACK_SMR_BAR - BBCI Feedback for SMR Training
%
%Synopsis:
% OPT= feedback_smr_bar(FIG, OPT, CTRL_FAST, CTRL_SLOW)
%
%Arguments:
% FIG  - handle of figure
% OPT  - struct of optional properties, see below
% CTRL_FAST - control signal to be received from the BBCI classifier 1
% CTRL_SLOW - control signal to be received from the BBCI classifier 2
%
%Output:
% OPT - updated structure of properties
%
%Optional Properties:

% Author(s): Benjamin Blankertz, Feb-2010

persistent H HH state memo cfd

if ~isstruct(opt) | ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  global VP_SCREEN
  opt.reset= 0;
  [opt, isdefault]= ...
      set_defaults(opt, ...
        'background', 0.5*[1 1 1], ...
        'color_bar', [0 0.6 0], ...
        'color_tri', [0 1 0], ...
        'bar_width', 0.3, ...
        'target_width', 0.075, ...
        'frame_color', 0.8*[1 1 1], ...
        'punchline_highlight_time', 1000, ...
        'punchline_spec', {'Color',[0 0 0], 'LineWidth',3}, ...
        'punchline_beaten_spec', {'Color',[1 1 0], 'LineWidth',5}, ...
        'gap_to_border', 0.02, ...
        'parPort', 1,...
        'log',1,...
        'fs', 25, ...
        'position', VP_SCREEN);
  
  [HH, cfd]= feedback_smr_bar_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);
  memo.punchline= 0;
  memo.punchline_counter= 0;
  do_set('init', handles, 'cursor_arrow', opt);
  do_set(200);
end

x1= ctrl_fast;
x1= max(0, min(1, x1));
x2= ctrl_slow;
x2= max(0, min(1, x2));

if x2>memo.punchline,
  YData= get(HH.punchline, 'YData');
  memo.punchline= x2;
  YData(:)= memo.punchline*2-1;
  do_set(H.punchline, 'YData',YData, opt.punchline_beaten_spec{:});
  memo.punchline_counter= opt.punchline_highlight_time/1000*opt.fs;
end
if memo.punchline_counter>0,
  memo.punchline_counter= memo.punchline_counter - 1;
  if memo.punchline_counter==0,
    do_set(H.punchline, opt.punchline_spec{:});
  end
end

YData= get(HH.bar, 'YData');
YData(3:4)= x2*2-1;
do_set(H.bar, 'YData',YData');
YData= get(HH.tri, 'YData');
YData(2)= max(x2*2-1, x1*2-1);
YData([1 3])= x2*2-1;
do_set(H.tri, 'YData',YData');
do_set('+');
