function opt = feedback_hexa(fig, opt, ctrl);
%FEEDBACK_HEXAWRITE - BBCI Typewriter Feedback with Hexagons
%
%Synopsis:
% OPT = feedback_hexawrite(FIG, OPT, CTRL)
%
%Arguments:
% FIG  - handle of figure
% OPT  - struct of optional properties, see below
% CTRL - control signal to be received from the BBCI classifier
%
%Output:
% OPT - updated structure of properties
%
%Optional Properties:
% degree_per_sec: rotation speed of the arrow
% grow_per_sec: maximum speed of elongation of the arrow (normalized,
%    1 means arrow grows from min to max in 1 sec)
% decay_per_sec: speed of shortning of the arrow (nromalized)
% threshold_move: arrow grows, when CTRL is above or equal this value
% threshold_turn: arrow rotates, when CTRL is below this value, choose
%    NaN for move-or-turn (i.e. threshold_turn=threshold_move).
% arrow_reset: determines to which direction the arrow is set after a select.
%    'none': no reset,
%    'bestletter': point to the hex that contains the most probable letter
%    'besthex': point to the hex with highest overall probability
% duration_show_selected: time to show selection in step 1, before
%  moving of the letter starts
% duration_show_final: time to show selection in step 2
% duration_move_selected: time to move small letters to hexagons
% duration_before_free: time after target presentation, before arrow 
%    starts moving
% countdown: length of countdown before application starts [ms]
% hexradius: radius of hexagons, normalized (YLim is [-1 1])
% label_radfactor: factor by which small letter are interior in the hexagons
% label_spec: text specification for small letters
% biglabel_spec: text specification for big letters
% biglabel_select_spec: text specification of selected letter
% msg_spec: text specification of messages ('pause', ...)
% text_spec: text specification of written text
% textfield_length: number of symbols shown in the text field
% arrow_width: width of the arrow normalized to hexdiameter
% arrow_backlength: length of arrow tail (going from center to rear side)
% arrow_minlength: length of arrow (going from center to pointing side)
% arrow_headlength: length of arrow head
% arrow_headwidth: width of arrow head
% arrow_spec: patch specification of arrow in normal operation
% arrow_select_spec: patch specification of arrow after hit
% hex_spec: line specification of hexagone outline
% hex_select_spec: line specification of selected hexagone outline
% background: background color
%
%Markers written to the parallel port
% 11-16: hex 1-6 was selected at step 1
% 21-26: hex 1-6 was selected at step 2
% 30-49: hex state changed
%  30: countdown starts
%  31: arrow turns and moves (hex selection, step 1)
%  32: hex at step 1 selected, unselected letters disappear
%  34: movement of small letters to hex positions starts
%  36: big letters appear
%  40: hex at step 2 selected, selected letter is marked
%  42: initial layout is shown again, pause before arrow movement starts
% 61-96: code of selected letter
% 200: init of the feedback
% 210: game status changed to 'play'
% 211: game status changed to 'pause'
% 212: game status changed to 'stop'
%
%See:
% tester_hexawrite, feedback_hexawrite_init

% Author(s): Benjamin Blankertz, Jan-2006

global DATA_DIR
persistent H HH state hex timer lm

if ~isfield(opt,'reset'),
  opt.reset = 1;
end

if opt.reset,
  opt.reset= 0;
  opt= set_defaults(opt, ...
		    'hexradius',0.29, ...
		    'labelset', ...
		    ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_<.?'],...
        'rate_control', 1, ...
        'speed', 2, ...
        'anglereset_factor', 0.5, ...
        'initangle_offset', pi/24, ...
        'duration_blocking', 500, ...
		    'duration_show_selected', 40, ...
		    'duration_show_final', 500, ...
		    'duration_move_selected', 500, ...
		    'duration_before_free', 600, ...
		    'countdown', 3000, ...
		    'arrow_reset', 'bestletter', ...
		    'degree_per_sec', 360/6, ...
		    'grow_per_sec', 1, ...
		    'decay_per_sec', 1, ...
		    'threshold_turn', -0.333, ...
		    'threshold_move', 0.333, ...
		    'threshold_preselect', 0, ...
		    'label_radfactor', 0.65, ...
		    'label_spec', ...
		      {'FontSize',0.06, 'FontWeight','bold'}, ...
		    'biglabel_spec', ...
		      {'Color','k','FontSize', 0.12}, ...
		    'biglabel_select_spec', ...
		      {'Color',[0 0.5625 0], 'FontSize',0.2}, ...
		    'msg_spec', {'FontSize',0.2}, ...
		    'text_spec', {'FontSize',0.06, 'FontName','Courier New', 'FontWeight','bold'}, ...
		    'textfield_width', 1.2, ...
        'textfield_to_hex_gap', 0.03, ...
        'textfield_to_top_gap', 0.01, ...
        'textfield_box_spec', {'Color',[0 0.5625 0], 'LineWidth',3}, ...
		    'hex_spec', {'Color','k', 'LineWidth',2}, ...
		    'hex_preselect_spec', {'LineWidth',4}, ...
		    'hex_select_spec', {'LineWidth',6}, ...
		    'language_model', 'german', ...
		    'lm_headfactor', [0.85 0.85 0.75 0.5 0.25], ...
		    'lm_letterfactor', 0.01, ...
		    'lm_npred', 2, ...
		    'lm_probdelete', 0.1, ...
		    'arcsteps', 10, ...
		    'arrow_width', 0.075, ...
		    'arrow_backlength', 0.1, ...
		    'arrow_minlength', 0.20, ...
		    'arrow_headlength', 0.15, ...
		    'arrow_headwidth', 0.175, ...
		    'arrow_spec', ...
		      {'FaceColor',[0.3 0.5 0.3], 'EdgeColor','none'},...
		    'arrow_grow_spec', ...
		      {'FaceColor',[0 0.7 0]}, ...
		    'arrow_select_spec', ...
		      {}, ...
        'background', 0.9*[1 1 1], ...
		    'fs', 25, ...
		    'text_reset', 0, ...
        'time_after_text_reset', 3000, ...
        'show_cake', 0, ...
        'show_control', 0, ...
        'control_x', [1.1 1.2], ...
        'control_h', 0.75, ...
        'control_meter_spec', {'FaceColor',[1 0.6 0]}, ...
        'log', 1, ...
		    'parPort', 1, ...
		    'status', 'pause', ...
        'changed', 1, ...
                    'cebit_layout', 1, ...
        'position', get(fig,'position'));

  if isnan(opt.threshold_turn),
    opt.threshold_turn= opt.threshold_move;
  end
  
  if isempty(opt.language_model),
    lm= [];
  else
    lm= lm_loadLanguageModel(opt.language_model);
    lm.charset= [lm.charset, '<'];
  end
  
  [HH, hex]= feedback_hexa_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);

  do_set('init',handles, 'hexa',opt);
  do_set(200);

  hex.biglabel_pos= zeros(6,2);
  for ii= 1:6,
    pos= get(HH.biglabel(ii), 'Position');
    hex.biglabel_pos(ii,:)= pos([1 2]);
  end
  hex.step= 1;
  hex.arrow_dir= 1;
  hex.arrow_len= 0; 
  hex.written= '';
  hex.written_ctrl= '';
  hex.codetable= opt.labelset';
  hex.codetable= [hex.codetable(:); ' '];
  hex.laststate= NaN;
  hex.lastdigit= NaN;
  hex.laststatus= 'urknall';
  hex.wasgrowing= 0;
  hex.preselect= 0;
  hex.pretarget= 6;
  hex.ctrl= 0;
  timer.msec= 0;
  timer.blocking= 0;
  hex= choose_start_angle(H, hex, lm, opt);
  state= -1;
end

if opt.changed,
  if ~strcmp(opt.status,hex.laststatus),
    hex.laststatus= opt.status;
    switch(opt.status),
     case 'play',
      do_set(210);
      state= 0;
     case 'pause',
      do_set(211);
      do_set(H.msg, 'String','pause', 'Visible','on');
      hex.arrow_len= 0;
      state= -1;
     case 'stop';
      do_set(212);
      do_set(H.msg, 'String','stopped', 'Visible','on');
      state= -2;
    end
  end
end

if opt.text_reset,
  opt.text_reset= 0;
  hex.written= '';
  do_set(H.textfield, 'String','');
  hex.written_ctrl= [hex.written_ctrl '|'];
  %% in case text_reset comes during movement, reset old positions
  for ii= 1:6,
    do_set(H.biglabel(ii), 'Position',hex.biglabel_pos(ii,:), ...
           'Visible','off');
  end
  opt.changed= 1;
  hex.step= 1;
  hex.arrow_len= 0;
  timer.msec= 0;
  opt.countdown= opt.time_after_text_reset;
  hex= choose_start_angle(H, hex, lm, opt);
  state= 0;
end

if opt.changed==1,
  opt.changed= 0;
  %% TODO (?): label_radfactor
  if opt.cebit_layout,
    do_set(H.logo, 'Visible','on');
  else
    do_set(H.logo, 'Visible','off');
  end
  if isnan(opt.threshold_turn),
    opt.threshold_turn= opt.threshold_move;
  end
  do_set(H.biglabel, opt.label_spec{:}, opt.biglabel_spec{:});
  do_set([H.hex.label], opt.label_spec{:});
  do_set([H.hex.outline], opt.hex_spec{:});
  do_set(H.textfield, opt.text_spec{:});
  do_set(H.msg, opt.msg_spec{:});
  do_set(H.arrow, opt.arrow_spec{:});
  if ~isnan(HH.control_threshold_turn),
    yy= -opt.control_h + (opt.threshold_turn+1)*opt.control_h;
    do_set(H.control_threshold_turn, 'YData', yy([1 1]));
    yy= -opt.control_h + (opt.threshold_move+1)*opt.control_h;
    do_set(H.control_threshold_move, 'YData', yy([1 1]));
    if opt.show_control,
      do_set([H.control_meter H.control_threshold_turn ...
              H.control_threshold_move H.control_outline'], 'Visible','on');
    else
      do_set([H.control_meter H.control_threshold_turn ...
              H.control_threshold_move H.control_outline'], 'Visible','off');
    end
  end
  if opt.show_cake,
    do_set(H.cake, 'Visible','on');
  else
    do_set(H.cake, 'Visible','off');
  end
  if ~isempty(opt.language_model),
    lm= lm_loadLanguageModel(opt.language_model);
    lm.charset= [lm.charset, '<'];
  end
  if strcmp(opt.status, 'play') & state~=0,
    do_set(H.msg, 'Visible','off');
  end
end

if opt.rate_control,
  if timer.blocking<=0, 
    hex.ctrl= hex.ctrl + ctrl*opt.speed/opt.fs;
    hex.ctrl= min(1, max(-1, hex.ctrl));
  else
    timer.blocking= timer.blocking - 1000/opt.fs;
  end
else
  hex.ctrl= ctrl;
end

if ~ismember(state, [3 4 5 11]),
  if hex.ctrl>=opt.threshold_move & ~hex.wasgrowing,
    hex.wasgrowing= 1;
    do_set(H.arrow, opt.arrow_grow_spec{:});
  elseif hex.ctrl<opt.threshold_move & hex.wasgrowing,
    hex.wasgrowing= 0;
    do_set(H.arrow, opt.arrow_spec{:});
  end
end

if state~=hex.laststate,
  hex.laststate= state;
  if state>=0,
    do_set(30+state);
  end
  fprintf('state: %d\n', state);
end

if state==0,
  if timer.msec>opt.countdown,
    do_set(H.msg, 'Visible','off');
    state= 1;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
  digit= ceil((opt.countdown-timer.msec)/1000);
  if digit~=hex.lastdigit,
    do_set(H.msg, 'String',int2str(digit), 'Visible','on');
    hex.lastdigit= digit;
  end
end

if state>=-1 & state<=1,
  if hex.ctrl<opt.threshold_turn,
    phi= hex.arrow_dir * opt.degree_per_sec/180*pi/opt.fs;
    hex.arrow_angle= mod( hex.arrow_angle + phi, 2*pi );
    pretarget= 1 + mod(ceil((hex.arrow_angle + 2*pi/12) / (2*pi/6))-1, 6);
    if opt.threshold_preselect==0 & pretarget~=hex.pretarget,
      hex.preselect= 1;
      do_set(H.hex(hex.pretarget).outline, opt.hex_spec{:});
      hex.pretarget= pretarget;
      do_set(H.hex(hex.pretarget).outline, opt.hex_preselect_spec{:});
    end
  end
end

if state==1,  %% choose target
  if hex.ctrl>=opt.threshold_move,
    hex.arrow_len= min(1, hex.arrow_len + opt.grow_per_sec/opt.fs);
    if hex.arrow_len>=1,
      hex.target= 1 + mod(ceil((hex.arrow_angle + 2*pi/12) / (2*pi/6))-1, 6);
      do_set(10*hex.step+hex.target);
      do_set(H.arrow, opt.arrow_select_spec{:});
      hex.preselect= 0;
      if opt.rate_control,
        hex.ctrl= opt.threshold_turn*(1-opt.anglereset_factor) + ...
                  opt.threshold_move*opt.anglereset_factor;
        timer.blocking= opt.duration_blocking;
      end
      if hex.step==1,
        state= 2;
      else
        state= 10;
      end
    else
      if hex.arrow_len>=opt.threshold_preselect & ~hex.preselect,
        hex.preselect= 1;
        hex.pretarget= 1 + mod(ceil((hex.arrow_angle+2*pi/12)/(2*pi/6))-1, 6);
        do_set(H.hex(hex.pretarget).outline, opt.hex_preselect_spec{:});
      end
    end
  else
    hex.arrow_len= max(0, hex.arrow_len - opt.decay_per_sec/opt.fs);
    if hex.arrow_len<opt.threshold_preselect & hex.preselect,
      hex.preselect= 0;
      do_set(H.hex(hex.pretarget).outline, opt.hex_spec{:});
    end
  end
end

if state==2,  %% target was selected at step 1
  timer.msec= 0;
  unselected= setdiff(1:6, hex.target);
  do_set([H.hex(unselected).label], 'Visible','Off');  %% time consuming
  do_set(H.hex(hex.target).outline, opt.hex_select_spec{:});
  state= 3;
end

if state==3,  %% show highlighted selected hex
  if timer.msec>opt.duration_show_selected,
    state= 4;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
end

if state==4,  %% initiate movement of selected
  timer.step= 1;
  nSteps= round(opt.duration_move_selected/1000*opt.fs) + 1;
  %% only steps 2:end of each path are used!
  hex.movepath_x= zeros(6, nSteps);
  hex.movepath_y= zeros(6, nSteps);
  shift= 1 + mod(3-hex.target-1, 6);
  for ii= 1:6,
    jj= 1 + mod(ii+shift-1, 6);
    ch= get(HH.hex(hex.target).label(jj), 'String');
    pos= get(HH.hex(hex.target).label(jj), 'Position');
    start_x= pos(1);
    start_y= pos(2);
    hex.movepath_x(ii,:)= linspace(start_x, hex.biglabel_pos(ii,1), nSteps);
    hex.movepath_y(ii,:)= linspace(start_y, hex.biglabel_pos(ii,2), nSteps);
    do_set(H.biglabel(ii), opt.label_spec{:}, 'String',ch, 'Visible','on');
  end
  do_set(H.hex(hex.target).label, 'Visible','off');
  i1= 1 + find(apply_cellwise2(opt.label_spec, 'strcmpi', 'FontSize'));
  i2= 1 + find(apply_cellwise2(opt.biglabel_spec, 'strcmpi', 'FontSize'));
  hex.fontsizepath= linspace(opt.label_spec{i1}, ...
                             opt.biglabel_spec{i2}, nSteps);
  hex.arrowlenpath= linspace(1, 0, nSteps);
  resetangle= (hex.target-1.5)*pi/3 + opt.initangle_offset;
  if abs(hex.arrow_angle-resetangle)>pi,  %% avoid going wrong direction
    resetangle= mod(resetangle, 2*pi);
  end
  hex.arrowanglepath= linspace(hex.arrow_angle, resetangle, nSteps);
  state= 5;
end

if state==5,  %% move select letters
  timer.step= timer.step + 1;
  if timer.step>size(hex.movepath_x,2),
    state= 6;
  else
    for ii= 1:6,
      new_x= hex.movepath_x(ii, timer.step);
      new_y= hex.movepath_y(ii, timer.step);
      do_set(H.biglabel(ii), 'Position',[new_x new_y 0], ...
             'FontSize',hex.fontsizepath(timer.step));
    end
    hex.arrow_len= hex.arrowlenpath(timer.step);
    hex.arrow_angle= hex.arrowanglepath(timer.step);
  end
end

if state==6,  %% prepare for step 2, show big letters
  if opt.duration_move_selected>0,
    do_set(H.biglabel, opt.biglabel_spec{:});
  end
  if opt.threshold_preselect==0,
    do_set(H.hex(hex.target).outline, opt.hex_preselect_spec{:});
  else
    do_set(H.hex(hex.target).outline, opt.hex_spec{:});
  end
  do_set(H.arrow, opt.arrow_spec{:});
  hex.wasgrowing= 0;
  hex.arrow_len= 0;
  timer.msec= 0;
  state= 7;
end

if state==7,
  if timer.msec>opt.duration_before_free,
    hex.step= 2;
    state= 1;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
end

if state==10,  %% selection at step 2
  written_char= get(HH.biglabel(hex.target), 'String');
  do_set(60+find(written_char==hex.codetable));
  if written_char~=' ',
    hex.written_ctrl= [hex.written_ctrl, written_char];
    fprintf('written: %s\n', hex.written_ctrl);
    if written_char=='<',
      hex.written= hex.written(1:max(0,end-1));
    else
      hex.written= [hex.written, written_char];
    end
    writ= [hex.written '_'];
    iBreaks= find(writ=='_');
    ll= 0;
    clear textstr;
    while length(iBreaks)>0,
      ll= ll+1;
      linebreak= iBreaks(max(find(iBreaks<hex.textfield_nChars)));
      if isempty(linebreak),
        %% word too long: insert hyphenation
        linebreak= hex.textfield_nChars;
        writ= [writ(1:linebreak-1) '-' writ(linebreak:end)];
      end
      textstr{ll}= writ(1:linebreak);
      writ(1:linebreak)= [];
      iBreaks= find(writ=='_');
    end
    textstr{end}= textstr{end}(1:end-1);
    textstr= textstr(max(1,end-hex.textfield_nLines+1):end);
    do_set(H.textfield, 'String',textstr);
  end
  hexprob= reorder_symbols(H, hex, lm, opt);
  switch(opt.arrow_reset),
   case 'none',
    hex.settohex= hex.target;
   case 'bestletter',
    [mm,mi]= max(hexprob(:));
    hex.settohex= ceil(mi/5);
   case 'besthex',
    [mm,hex.settohex]= max(sum(hexprob,1));
   otherwise,
    error('policy for arrow_reset not known');
  end
  hex.arrow_angle= (hex.settohex-1.5)*pi/3 + opt.initangle_offset;
  if opt.threshold_preselect==0,
      do_set(H.hex(hex.pretarget).outline, opt.hex_spec{:});
      hex.preselect= 1;
      hex.pretarget= hex.settohex;
      do_set(H.hex(hex.settohex).outline, opt.hex_preselect_spec{:});
  end
  do_set(H.biglabel(hex.target), opt.biglabel_select_spec{:});
  do_set(H.biglabel(setdiff(1:6,hex.target)), 'Visible','off');
%  do_set(H.hex(hex.target).outline, opt.hex_select_spec{:});
  do_set(H.arrow, opt.arrow_spec{:});
  hex.wasgrowing= 0;
  hex.arrow_len= 0;
  timer.msec= 0;
  state= 11;
end

if state==11,
  if timer.msec>opt.duration_show_final,
%    do_set(H.hex(hex.target).outline, opt.hex_spec{:});
    do_set(H.biglabel(setdiff(1:6,hex.target)), opt.biglabel_spec{:});
    do_set(H.biglabel(hex.target), opt.biglabel_spec{:}, 'Visible','off');
    do_set([H.hex.label], 'Visible','on');
    timer.msec= 0;
    state= 12;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
end

if state==12,
  if timer.msec>opt.duration_before_free,
    hex.step= 1;
    state= 1;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
end

if ~isnan(HH.control_meter),
  yy= -opt.control_h + (hex.ctrl+1)*opt.control_h;
  do_set(H.control_meter, 'YData', [-opt.control_h*[1 1] yy*[1 1]]);
end

arrow= fb_hexawrite_arrow(hex.arrow_angle, hex.arrow_len, opt);
do_set(H.arrow, 'XData',hex.ursprung(1)+arrow(1,:)', ...
       'YData',hex.ursprung(2)+arrow(2,:)');

%% for debugging purpose
%opt.hex= hex;
%opt.state= state;
%opt.timer= timer;
%opt.lm= lm;

do_set('+');
return;





function hex= choose_start_angle(H, hex, lm, opt)

%% in the beginning, delete does not make sense, so set lm_probdelete to 0
hexprob= reorder_symbols(H, hex, lm, setfield(opt,'lm_probdelete',0));
switch(opt.arrow_reset),
 case 'none',
  settohex= 1;
 case 'bestletter',
  [mm,mi]= max(hexprob(:));
  settohex= ceil(mi/5);
 case 'besthex',
  [mm,settohex]= max(sum(hexprob,1));
 otherwise,
  error('policy for arrow_reset not known');
end
hex.arrow_angle= (settohex-1.5)*pi/3 + opt.initangle_offset;
return;

  
  
  

function hexprob= reorder_symbols(H, hex, lm, opt)

hexprob= zeros(5,6);
if isempty(lm),
  hexprob(:)= 1/30;
  do_set([H.hex.label], 'Visible','on');
  return;
end

prob= lm_getProbability(lm, hex.written, opt);

rankpos= [3 4 5 1 2];
for hi= 1:6,
  lab= opt.labelset(hi,:);
  for li= 1:5,
    ii= find(lm.charset==lab(li));
    hexprob(li,hi)= prob(ii);
  end
  [so,si]= sort(-hexprob(:,hi));
  for li= 1:5,
    do_set(H.hex(hi).label(rankpos(li)), 'String',lab(si(li)));
  end
end
return;
