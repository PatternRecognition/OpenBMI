function fb_opt = feedback_speller_2d(fig, fb_opt, x, y);

persistent status counter active cross pat ...
    cou seqpos target hit miss ht fbb target_rect h_reg time0 midpos ...
    fid lastbeaten xpos ypos timeoutcounter stopped pause_after_timeout pause_counter sound_files break_counter additionalbreak

persistent state cursor paused sequence 
persistent moveIdx moveStep
persistent h_cursor h_cross h_text free_balls next_target
persistent h_boxtext_cell h_line_cell h_rect_cell
persistent h_rect h_line h_boxtext T tree_ind buffertext
persistent boxnum rollbackcount handle h_buffertext rev_ind

if ~exist('y','var') | isempty(y)
    y = 0;
end

gray = [1,1,1]*0.8;
GRAY = [1,1,1]*0.8;
blue = [0 0 1];
green = [0 1 0];
red = [1 0 0];

false= 0;  %% for R<13
true= 1; %% for R<13

if isempty(fb_opt)
  fb_opt = struct('reset',1);
end

if ~isfield(fb_opt,'reset') 
  fb_opt.reset = 1;
end

neustart = 0;

if fb_opt.reset,
  neustart = 1;
  fb_opt= set_defaults(fb_opt, ...
		'alwaysreverse',0,...
		'backspace', 50,...
        'background_color', 0.9*[1 1 1], ...
        'break',[inf 0],...
		'boxtext_fontsize', 0.06, ...
        'boxtext_mid', [-.9 .9],...
		'buffertext_pos', [0 -0.8], ...
		'buffertext_fontsize', 0.06, ...
        'buffertext','',...
        'buffertext_len',20,...
		'changed',0,...
		'cone', 0, ...
		'countdown', 5000, ...
		'damping', 10, ...
		'fixation_color', 0.3*[1 1 1], ...
		'fixation_size', 0.075, ...
		'fixation_linewidth', 3, ...
		'fixation_position', [0 0], ...
		'fix_lines',0,...
		'free_region', .4, ...
		'free_balls',0,...
		'fs', 25, ...
		'gradient',inf,...
		'log', 0, ...
		'marker_active', 'r+', ...
		'marker_nonactive', 'k.', ...
		'marker_active_size', [60 15], ...
		'marker_nonactive_size', [50 5], ...
		'middleposition',0,...
		'msg_fontsize', 0.15, ...
		'nontarget_color',gray,...
		'orientation', 'landscape', ...
		'parPort', 1,...
		'pause_after_timeout',0,...
		'position', get(fig,'position'),...
		'relational',0,...
        'reset',0, ...
	    'rollback', 0,...
		'status', 'pause', ...
		'target_mode', 1, ...
		'target_dist', 0.1, ...
		'target_width', 0.05, ...
		'targets', 2, ...
		'target_height', 0.1, ...
        'text_reset',0,...
		'time_after_hit', 500, ...
        'time_after_letter',0,...
        'time_after_text_reset',2000,...
		'time_before_free', 500, ...
		'target_color', [0 0.7 0], ...
        'tree',T,...
        'trial_duration', 2000);
%%%%%%%
         
   
  paused= 0;
  state= 0;
  
  %initialize tree and tree-position
  T = generate_tree(fb_opt);
  
  fb_opt.tree = T;
  tree_ind = 1;

  rollbackcount = 0;
  boxnum = fb_opt.targets; % number of boxes which are currently shown.
  buffertext = fb_opt.buffertext(max(1,length(fb_opt.buffertext)-fb_opt.buffertext_len+1):end);
  
 %%%%%%%%%%%%%%
  
  active = false;
  status = 0;
  counter = ceil(fb_opt.countdown*fb_opt.fs/1000)+1;
  fb_opt.reset = 0;
  seqpos = 1;
  hit = 0;
  miss = 0;
  fbb= fb_opt;
  fbb.status = '';
  midpos = 0;

  [handle,other_args] = feedback_speller_2d_init(fig,fb_opt);
  % pass the right handles.
  cross = 3; 
  cou = 4; 
  pat = 5; 
  h_buffertext= 6; 
  text = 7; 
  h_boxtext = 8:13;%HACK
  h_reg = 14:15;%HACK
   % These are the side-results of feedback_speller_2d_init:
   target_rect = other_args{1};
       
  do_set('init',handle,'speller_2d',fb_opt);
  do_set(200);
  pause_after_timeout = 0;
  fb_opt.changed = 0;
  stopped = 0;
end


if fb_opt.changed == 1
   
    
  if fbb.free_region~=fb_opt.free_region,
    if fb_opt.free_region>=1,
      do_set(pat,'Visible','off');
    else
      do_set(pat, 'XData',[-1 1 1 -1]*fb_opt.free_region, ...
             'YData',[-1 -1 1 1]*fb_opt.free_region, ...
             'Visible','on');
    end
  end
  
  if active,
    if any(fbb.marker_active_size~=fb_opt.marker_active_size)
      do_set(cross,'MarkerSize',fb_opt.marker_active_size(1), ...
                'LineWidth',fb_opt.marker_active_size(2));
    end
    if ~strcmp(fbb.marker_active,fb_opt.marker_active)
      do_set(cross,'Marker',fb_opt.marker_active(2), ...
                'Color',fb_opt.marker_active(1));
    end
  else
    if any(fbb.marker_nonactive_size~=fb_opt.marker_nonactive_size)
      do_set(cross,'MarkerSize',fb_opt.marker_nonactive_size(1), ...
                'LineWidth',fb_opt.marker_nonactive_size(2));
    end
    if ~strcmp(fbb.marker_nonactive,fb_opt.marker_nonactive)
      do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
                'Color',fb_opt.marker_nonactive(1));
    end
  end
end



if fb_opt.changed | neustart   
  if ~strcmp(fbb.status,fb_opt.status)
    switch fb_opt.status
     case 'play'
      do_set(210);
     case 'pause'
       if pause_after_timeout==0
         do_set(211);
       end
     case 'stop'
      do_set(212);
      stopped = 1;
    end
    switch fb_opt.status
     case 'play'
       if stopped
         stopped = 0;
          status = 0;
         counter = ceil(fb_opt.countdown*fb_opt.fs/1000)+1;
         seqpos = 1;
       end
      if status>0
        do_set(cou,'Visible','off');
      end 
     case  'stop'
      fbb.status = 'stop';
    end
  end
end


fb_opt.changed = 0;

  

if strcmp(fb_opt.status, 'stop')
  do_set('+');
  return;
end

if fb_opt.text_reset
    status = 1;
    counter = ceil((fb_opt.time_after_text_reset+fb_opt.time_after_hit)*fb_opt.fs/1000)+1;
    tree_ind = 1;
    buffertext = fb_opt.buffertext;
    rollbackcount = 0;
    do_set(h_buffertext,'String',buffertext);
    do_set(h_boxtext,'Visible','off');
    fb_opt.text_reset = 0;
end 

%%%% Move the cursor:
if status<3 & (fb_opt.relational>0 | fb_opt.middleposition>0)
  x = 0;
  y = 0;
elseif (fb_opt.relational>0 | fb_opt.middleposition>0)
  if fb_opt.middleposition &midpos>0
    x = (fb_opt.middleposition-midpos)/fb_opt.middleposition*x;
    y=  (fb_opt.middleposition-midpos)/fb_opt.middleposition*y;
    midpos = midpos-1;
  end
  
  if fb_opt.relational==1
    x = x/fb_opt.damping+xpos;
    y = y/fb_opt.damping+ypos;
  else
    x = x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping)+xpos*fb_opt.relational;
    y = y*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping)+ypos*fb_opt.relational;
  end
end

if abs(x)>1
  y = y/abs(x);
  x = x/abs(x);
end
if abs(y)>1
  x = x/abs(y);
  y = y/abs(y);
end

if x-xpos>fb_opt.gradient
  x = xpos+fb_opt.gradient;
elseif xpos-x>fb_opt.gradient
  x = xpos-fb_opt.gradient;
end

if y-ypos>fb_opt.gradient
  y = ypos+fb_opt.gradient;
elseif ypos-y>fb_opt.gradient
  y = ypos-fb_opt.gradient;
end

if fb_opt.fix_lines
    % modify x and y to fix lines
    switch fb_opt.target_mode
        case 1
            y = 0;
        case 2
            x = 0;
    end
end

do_set(cross, 'XData',x, 'YData',y); 

%%% up to here: cursor has been moved. Now comes the gaming logic.
if strcmp(fb_opt.status,'pause')
  if pause_after_timeout==0
    do_set(cou,'String','pause', 'Visible','on');
  else
    pause_counter = pause_counter-1;
    do_set(cou,'String',int2str(ceil(pause_counter/fb_opt.fs)), 'Visible','on');
    if pause_counter<=0
      fb_opt.status = 'play';
      fb_opt.changed = 1;
      do_set(250);
      do_set(h_reg(target),'FaceColor',blue);
      pause_after_timeout = 0;
      status = 3;
      do_set(cou,'Visible','off');
    end
  end
  do_set('+');
  return;
else
if status==0,
    % countdown before the game.
  counter = counter-1;
  if counter<=0
    do_set(cou,'Visible','off');
    status = 1;
    counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000)+1;
    do_set(201);
    
    additionalbreak = fb_opt.time_after_letter;
    break_counter = 0;    
    
  else
    do_set(cou,'String',int2str(ceil(counter/fb_opt.fs)));
  end
end

if status==1,
    % countdown finished. initialize graphics.
  lastbeaten = 0;
  counter = counter-1;
  if counter<=0
      for i = 1:length(h_reg)
        do_set(h_reg(i),'FaceColor',fb_opt.nontarget_color);
      end
      
      if break_counter<0
        do_set(h_boxtext,'visible','off');
        do_set(cross,'Visible','off');
        do_set(cou,'String',ceil(abs(break_counter)/1000),'Visible','on');
        break_counter = break_counter + 1000/fb_opt.fs;
        if break_counter>=0
            break_counter = 0;
            do_set(cross,'Visible','on');
            do_set(cou,'Visible','off');
        end 
        do_set('+');
        return;
      end
      
      
    status = 2;
    counter = ceil((additionalbreak+fb_opt.time_before_free)*fb_opt.fs/1000);
    
    %% arrange boxes and boxtext.
    %do_set([h_boxtext],'visible','off');
    %do_set(h_boxtext,'String','');
   if fb_opt.rollback==0|(fb_opt.rollback>max(0,rollbackcount))...
        |(fb_opt.rollback==-1&(~isempty(T(tree_ind).parent)|rollbackcount==0))
    
    % show letter selection.
    if isempty(T(tree_ind).parent)
      % if the last character is a <, remove chars.
      if length(buffertext)>0&buffertext(end)=='<'
        buffertext = buffertext(1:(length(buffertext)-2));
        do_set(h_buffertext,'String',untex(buffertext));
      end
    end
    
    do_set([h_boxtext],'visible','on');
    box_ind = 1:length(T(tree_ind).children);
    
    for i = box_ind
        str = get_boxlabel(T(T(tree_ind).children(i)).leaves);
        for j = 1:3
            do_set(h_boxtext((i-1)*3+j),'String',str{j});
        end 
     % do_set(h_boxtext(i),...
      %       'String',untex(get_boxlabel(T(T(tree_ind).children(i)).leaves)));
    end
    do_set(100+tree_ind);
  else
    % show confirm selection.
    boxnum = 2;
    do_set([h_boxtext],'visible','on');
    do_set(h_boxtext(1),'String','right');
    do_set(h_boxtext(2),'String','wrong');
    do_set(-10);
  end  
   end
end

if status==2,
    %free gaming before releasing the cursor
  counter = counter-1;
  if counter<=0
    status = 3;
  end
end



if status==3,
  % wait before activating the cursor.
  if (fb_opt.free_region>=1) | (max(abs(x),abs(y))<fb_opt.free_region)  | ((abs(x)<fb_opt.free_region) & ((y*ypos)<=0)) ...
        | ((abs(y)<fb_opt.free_region) & ((x*xpos)<=0)) | (((x*xpos)<=0) & ((y*ypos)<=0))
    status = 4;
    active = true;
    do_set(60);
    do_set(cross,'MarkerSize',fb_opt.marker_active_size(1), ...
           'LineWidth',fb_opt.marker_active_size(2), ...
           'Marker',fb_opt.marker_active(2), ...
           'Color',fb_opt.marker_active(1));
  end
end

if status==4,
	% wait until something is hit.
	touch= pointinrect([x y], target_rect);
	%keyboard
	if ~isempty(touch),
            do_set(h_reg(touch), 'FaceColor',green);
            do_set(30+touch);
		status = 1;
		do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
               'Color',fb_opt.marker_nonactive(1), ...
               'MarkerSize',fb_opt.marker_nonactive_size(1), ...
               'LineWidth',fb_opt.marker_nonactive_size(2));
		do_set(setdiff(h_reg, h_reg(touch)), ...
           'faceColor',fb_opt.nontarget_color);
		
		counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
		active = false;
		
		%%% now transmit the information of the choice to the speller tree.
		if fb_opt.rollback~=0
		    rollbackcount = rollbackcount+1;
		end
        additionalbreak = 0;
		if strcmp(get(handle(h_boxtext(touch)),'String'),'wrong')
		if fb_opt.rollback>0
		[tree_ind,buffertext] = reverse_steps(T,tree_ind,buffertext,fb_opt.rollback);
		else
		% fb_opt.rollback is negative. start over the whole letter.
		tree_ind = 1;
        additionalbreak = fb_opt.time_after_letter;
        break_counter = break_counter+1;
        if break_counter>=fb_opt.break(1)
            break_counter = -fb_opt.break(2);
        end 
		buffertext = buffertext(1:end-1);
		end
		rollbackcount = 0;
		elseif strcmp(get(handle(h_boxtext(touch)),'String'),'right')
		% do nothing, just reset the counter.
		if fb_opt.rollback~=0
		rollbackcount = 0;
		end
		elseif length(T(T(tree_ind).children(touch)).leaves)==1
		% arrived at a leaf.
        
        % sendet code zum ParallelPort über den Buchstraben
        do_set(strfind(T(1).leaves,T(T(tree_ind).children(touch)).leaves));
                
		buffertext = [buffertext ...
                    T(T(tree_ind).children(touch)).leaves];
		tree_ind = 1;
        additionalbreak = fb_opt.time_after_letter;
        break_counter = break_counter+1;
        if break_counter>=fb_opt.break(1)
            break_counter = -fb_opt.break(2);
        end 
		else
		% normal case: proceed in the tree.
		tree_ind = T(tree_ind).children(touch);
		end
		counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000)+1;
		end
		do_set(h_buffertext,'String',untex(buffertext(max(1,(length(buffertext)-fb_opt.buffertext_len)):end)));
	end
end




xpos = x; ypos = y;

do_set('+');

return

function str = get_boxlabel(str_in)
% get boxlabels from string. 'str' contains all letters in the box.
if length(str_in)>1
  %str = [str(1),'...',str(end)];
  str{1} = untex(str_in(1));
  str{2} = '...';
  str{3} = untex(str_in(end));
else
    str{1} = '';
    str{2} = untex(str_in);
    str{3} = '';
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
