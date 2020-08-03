function opt = feedback_speller_basket(fig, opt, control,dum);

persistent state cursor target paused sequence h_rect2
persistent basket running_time waiting moveIdx moveStep
persistent h_cursor h_cross h_text free_balls next_target
persistent h_boxtext_cell h_line_cell h_rect_cell
persistent h_rect h_line h_boxtext T tree_ind buffertext
persistent boxnum rollbackcount handle h_buffertext rev_ind
persistent alwaysreverse

global lost_packages

if ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt= set_defaults(opt, ...
                    'trial_duration', 2000, ...
                    'targets', 4, ...
                    'target_height', 0.1, ...
                    'outerTarget_size', 0.5, ...
                    'orientation', 'landscape', ...
                    'direction', 'downward', ...
                    'relative', 0, ...
                    'boxtext_fontsize', 0.06, ...
                    'cone', 0, ...
                    'cursor_size', 30, ...
                    'cursor_type', 'o', ...
                    'cursor_color', [0.8 0 0.8], ...
                    'buffertext_pos', [-0.86 -0.9], ...
                    'buffertext_fontsize', 0.06, ...
                    'msg_fontsize', 0.15, ...
                    'show_counter', 'on', ...
                    'show_fixation', 'off', ...
                    'background_color', 0.9*[1 1 1], ...
                    'target_color', [0 0.7 0], ...
                    'nontarget_color', 0.4*[1 1 1], ...
                    'hit_color', [0 0.9 0], ...
                    'miss_color', [1 0 0], ...
                    'fixation_color', 0.3*[1 1 1], ...
                    'fixation_size', 0.075, ...
                    'fixation_linewidth', 3, ...
                    'fixation_position', [0 0], ...
                    'free_balls',0,...
                    'time_after_hit', 250, ...
                    'time_before_next', 0, ...
                    'time_before_free', 750, ...
                    'countdown', 5000, ...
                    'matchpoints', 100, ...
                    'show_points', 'on', ...
                    'ms', 40, ...
                    'balanced_sequence',0,...
                    'damping', 10, ...
                    'status', 'pause', ...
                    'log', 0, ...
                    'changed', 0, ...
                    'position', get(fig,'position'),...
                    'parPort',1, ...
                    'rollback', 0,...
                    'alwaysreverse',0,...
                    'backspace', 50);

  [handle,other_args]=feedback_speller_basket_init(fig,opt);
  do_set('init',handle,'basket',opt);

  
  % These are the handle indices according to feedback_basket_init:
  h_cursor = 3;
  h_cross = 4:5;
  h_buffertext = 6;
  h_text = 7;
  
  % the following cell arrays contain handle arrays for each number of boxes.
  h_rect_cell = cell(1,opt.targets-1);
  h_boxtext_cell = cell(1,opt.targets-1);
  h_line_cell = cell(1,opt.targets-1);
  for i = 2:opt.targets
    h_rect_cell{i-1} = (7+(i-1)*i/2-1)+(1:i);
    h_boxtext_cell{i-1} = (7+opt.targets*(opt.targets+1)/2-1+(i-1)*i/2-1)+(1:i);
    h_line_cell{i-1} = (7+opt.targets*(opt.targets+1)-2+(i-2)*(i-1)/2)+(1:i-1);
  end
  % initialize the boxes with the maximal number:
  h_rect = h_rect_cell{end};
  h_boxtext = h_boxtext_cell{end};
  h_line = h_line_cell{end};
  % These are the side-results of feedback_basket_init:
  basket = other_args{1};
  moveIdx = other_args{2};
  
  cursor= [0 0];
  cursor(moveIdx)= -1;
  moveStep= 2/(opt.trial_duration/opt.ms);

  opt.reset = 0;
  waiting= 0;
  paused= 0;
  state= 0;
  if opt.free_balls>0
    free_balls=opt.free_balls;
  else
    free_balls=-1;
  end
  next_target=[];
  target = [];
  
  %initialize tree and tree-position
  T = generate_tree(opt);
  tree_ind = 1;

  rollbackcount = 0;
  boxnum = opt.targets+sign(opt.alwaysreverse); % number of boxes which are currently shown.
  buffertext = '';
  
  rev_ind=opt.targets+2;
  alwaysreverse = opt.alwaysreverse;  
  end

%keyboard
if free_balls==0
  free_balls = -1;
end

if opt.changed==1 & opt.log==1,
  opt.changed= 0;
  % write log?
end

if strcmp(opt.status, 'pause') & ~paused,
  do_set(h_text, 'string','pause', 'visible','on');
  do_set(211);
  paused= 1;
end

if paused,
  if strcmp(opt.status, 'play'),
    do_set(209);
    waiting= 0;
    paused= 0;
    state= 0;
  else
    do_set('+');
    return;
  end
end

switch(state),
 case 0,  %% countdown to start
  waiting= waiting + opt.ms;
  if waiting>=opt.countdown,
    do_set(h_text, 'visible','off');
    do_set(210);
    running_time= 0;
    state= 1;
  else
    str= sprintf('start in %.0f s', ceil((opt.countdown-waiting)/1000));
    do_set(h_text, 'string',str, 'visible','on');
  end
 case 1, %% arrange boxes and boxtext.
  do_set([h_rect,h_line,h_boxtext],'visible','off');
  do_set(h_boxtext,'String','');
  if opt.rollback==0|(opt.rollback>max(0,rollbackcount))...
        |(opt.rollback==-1&(~isempty(T(tree_ind).parent)|rollbackcount==0))
    % show letter selection.
    if isempty(T(tree_ind).parent)
      % if the last character is a <, remove chars.
      if length(buffertext)>0&buffertext(end)=='<'
        buffertext = buffertext(1:(length(buffertext)-2));
      end
    end
    boxnum = length(T(tree_ind).children)+sign(alwaysreverse);
    h_rect = h_rect_cell{boxnum-1};
    h_line = h_line_cell{boxnum-1};
    h_boxtext = h_boxtext_cell{boxnum-1};
    do_set([h_rect,h_line,h_boxtext],'visible','on');
    box_ind = 1:length(T(tree_ind).children);
    if sign(alwaysreverse)
      rev_ind = min(alwaysreverse,boxnum);
      do_set(h_boxtext(rev_ind),'String','rev');
    end
    for i = box_ind
      do_set(h_boxtext(i+(rev_ind<=i)),...
             'String',untex(get_boxlabel(T(T(tree_ind).children(i)).leaves)));
    end
    do_set(tree_ind);
  else
    % show confirm selection.
    boxnum = 2;
    h_rect = h_rect_cell{boxnum-1};
    h_line = h_line_cell{boxnum-1};
    h_boxtext = h_boxtext_cell{boxnum-1};
    do_set([h_rect,h_line,h_boxtext],'visible','on');
    do_set(h_boxtext(1),'String','right');
    do_set(h_boxtext(2),'String','wrong');
    do_set(-10);
  end  
  waiting= opt.time_before_free;
  state= 5;
 case 2,  %% wait for target hit
  cursor(moveIdx)= cursor(moveIdx) + moveStep;
  touch= pointinrect(cursor, basket{boxnum-1});
  if ~isempty(touch),
    if free_balls>0
      free_balls=free_balls-1;
    end
    do_set(-touch);
    do_set(h_rect(touch), 'faceColor',opt.hit_color);   
    do_set(setdiff(h_rect, h_rect(touch)), ...
           'faceColor',opt.nontarget_color);
    waiting= opt.time_after_hit;
    state= 3;
    % now consider the next state in the tree:
    if opt.rollback~=0
      rollbackcount = rollbackcount+1;
    end
    touchisbigger=(touch>=rev_ind);%&(rev_ind<opt.targets+2);
    if strcmp(get(handle(h_boxtext(touch)),'String'),'wrong')
      if opt.rollback>0
        [tree_ind,buffertext] = reverse_steps(T,tree_ind,buffertext,opt.rollback);
      else
        % opt.rollback is negative. start over the whole letter.
        tree_ind = 1;
        buffertext = buffertext(1:end-1);
      end
      rollbackcount = 0;
    elseif strcmp(get(handle(h_boxtext(touch)),'String'),'right')
      % do nothing, just reset the counter.
      if opt.rollback~=0
        rollbackcount = 0;
      end
    elseif strcmp(get(handle(h_boxtext(touch)),'String'),'rev')
       [tree_ind,buffertext] = reverse_steps(T,tree_ind,buffertext,1);     
       if opt.rollback~=0
         rollbackcount = rollbackcount-1;
       end
    elseif length(T(T(tree_ind).children(touch-touchisbigger)).leaves)==1
      % arrived at a leaf.
      buffertext = [buffertext ...
                    T(T(tree_ind).children(touch-touchisbigger)).leaves];
      tree_ind = 1;
    else
      % normal case: proceed in the tree.
      tree_ind = T(tree_ind).children(touch-touchisbigger);
    end
  end
  do_set(h_buffertext,'String',untex(buffertext(max(1,(length(buffertext)-20)):end)));
 case 3,  %% wait after target hit
  waiting= waiting - opt.ms;
  if waiting<0, 
    do_set(h_rect, 'faceColor',opt.nontarget_color);
    cursor(moveIdx)= -1;
    if opt.relative,
      cursor(3-moveIdx)= 0;
    end
    waiting= opt.time_before_next;
    if waiting>0,
      state= 4;
    else
      state= 1;
    end
  end
 case 4,  %% wait before showing next choice
  waiting= waiting - opt.ms;
  if waiting<0,
    state= 1;
  end
 case 5,  %% wait before freeing the cursor
  waiting= waiting - opt.ms;
  if waiting<0,
    state= 2;
  end
end

if state>0,
  running_time= running_time + opt.ms*(1+lost_packages);
end

if state==1 | state==2 | ...
    (~opt.relative & (state==3 | state==4)),
  if opt.relative,
    cursor(3-moveIdx)= cursor(3-moveIdx) + control/opt.damping;
  else
    cursor(3-moveIdx)= control;
  end
  cursor(3-moveIdx)= max(-1, min(1, cursor(3-moveIdx)));
end

cc= cursor;
if opt.cone>0,
  cf= min(1, (cursor(moveIdx)+1)/2/opt.cone);
  cc(3-moveIdx)= cf * cursor(3-moveIdx);
end

if state==2
  do_set(h_cursor, 'xData',cc(1), 'yData',cc(2));
end

do_set('+');


function str = get_boxlabel(str)
% get boxlabels from string. 'str' contains all letters in the box.
if length(str)>2
  str = [str(1),'...',str(end)];
end
return

function [tree_ind,buf] = reverse_steps(T,tree_ind,buf,rollback_num)
% go back rollback_num decision steps. If letters have to be erased, 
% shorten the string buf.
for i = 1:rollback_num
  if isempty(T(tree_ind).parent)
    % root of the tree.
    if isempty(buf)
      % beginning. no more reversal possible. do nothing.
    else
      % shorten buf and go to the parent of the last leaf.
      l = buf(end);
      buf(end) = [];
      s = strvcat(T.leaves);
      tree_ind = T(max(find(s(:,1)==l))).parent;
    end
  else
    % not the root - go to the parent.
    tree_ind = T(tree_ind).parent;
  end
end
return