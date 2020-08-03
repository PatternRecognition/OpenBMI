function fb_opt= animate_fb_speller(fb_opt, ptr, dscr_out, dtct_out, comb_out,error_out)
%fb_opt= animate_fb_speller(fb_opt, ptr, dscr_out, dtct_out,
%comb_out,error_out)

global BCI_DIR mrk pp first_test_event time_line trigger

if isempty(trigger)
  trigger = cell(2,2);
  trigger{1,1} = 'L';
  trigger{1,2} = [1,65,70];
  trigger{2,1} = 'R';
  trigger{2,2} = [2,74,192];
end

colors = [1 0 0;0 1 0; 0 0 1];
persistent lettertree letterleft letterright pointer ...
    status word ha1 ha2left ha2right ha3 ht_marker

MARKER_VIEW= 3;

if isequal(ptr,'init')
  status = 0;
  pointer = 1;
  word = '';

  if ~isfield(fb_opt,'language')
    fb_opt.language = 'deAlphaProb.tab';
  end
  
  [letters,probs] = textread([BCI_DIR 'pseudo_online/' fb_opt.language],'%s%f',-1);
  lettertree = create_lettertree(letters,probs,1,[]);
  
  clf;
  ha1= subplot('position', [0 .01 .999 .09]); box on;
  set(gca, 'xTick',0, 'xTickLabel','', 'xLim',[-1000 1000]*MARKER_VIEW, ...
           'yTick',[], 'yLim',[-1 1]);
  maxMarkers= 10*MARKER_VIEW;           %% 5 markers per second is maximum
  ht_marker= text(NaN*ones(1,maxMarkers), zeros(1,maxMarkers), '');
  set(ht_marker, 'fontSize',28, 'fontWeight','bold');
  
  let=max(length(lettertree(pointer).left),length(lettertree(pointer).right));
  col = floor(sqrt(let));
  letterleft = sprintf([repmat('%s',[1 col]) '\n'], ...
		       lettertree(pointer).left{:});
  letterright = sprintf([repmat('%s',[1 col]) '\n'], ...
		       lettertree(pointer).right{:});

  ha3 = subplot('position',[0 0.85 1 0.1]);box on;
  set(ha3, 'xTick',[], 'yTick',[]);
  s = text(0.1,0.5,untex(word));
  if isfield(fb_opt,'fontSize_words')
    set(s,'FontSize',fb_opt.fontSize_words);
  end
  
  ha2left= subplot('position', [0.01 .11 0.49 .72]);box on;
  set(ha2left, 'xTick',[],'yTick',[]);
 
  set(ha2left,'Color',[0.8 0.8 0.8]);
  s = text(0.1,0.5,untex(letterleft));
  if isfield(fb_opt,'fontSize')
    set(s,'FontSize',fb_opt.fontSize);
  end
  
  ha2right= subplot('position', [0.51 .11 0.48 .72]); box on;
  set(ha2right, 'xTick',[],'yTick',[]);
  set(ha2right,'Color',[0.8 0.8 0.8]);
  
  s = text(0.1,0.5,untex(letterright));
  if isfield(fb_opt,'fontSize')
    set(s,'FontSize',fb_opt.fontSize);
  end
    
  
  drawnow
else
  switch status 
    case 0 % waiting
    if comb_out(ptr)==0
      % nothing to do!!!
    else
      if comb_out(ptr)<0
	set(ha2left,'Color',[1 0 0]);
      else
	set(ha2right,'Color',[1 0 0]);
      end	
      status = comb_out(ptr);
      drawnow
    end
   case {-1, 1}  % waiting for confirmation
    if error_out(ptr)~=0 % otherwise is nothing to do
      if error_out(ptr)==2 & isfield(fb_opt,'stepBack') & fb_opt.stepBack==0
	status = -status;
	error_out(ptr)=1;
      end
      switch error_out(ptr)
       case 2 % rejection
	set(ha2left,'Color',[1 1 1]*0.8);
	set(ha2right,'Color',[1 1 1]*0.8);
	drawnow
	status = 0;
       case 1 %acceptation
	if status==-1, 
	  pointer = lettertree(pointer).leftchild; 
	else
	  pointer = lettertree(pointer).rightchild;
	end
	if isempty(pointer)
	  if status==-1, letter = letterleft(1);
	  else letter=letterright(1); 
	  end
	  word = [word letter];
	  if isfield(fb_opt,'word_history')
	    word = word(max(1,length(word)- ...
			    fb_opt.word_history+1):end);
	  end
	  
	  ch = get(ha3,'Children');
	  set(ch,'String',untex(word));
	  
	  pointer = 1;
	end
	
	let=max(length(lettertree(pointer).left),length(lettertree(pointer).right));
	col = floor(sqrt(let));
	letterleft = sprintf([repmat('%s',[1 col]) '\n'], ...
			     lettertree(pointer).left{:});
	letterright = sprintf([repmat('%s',[1 col]) '\n'], ...
			      lettertree(pointer).right{:});
	
	set(ha2left,'Color',[1 1 1]*0.8);
	ch = get(ha2left,'Children');
	set(ch,'String',untex(letterleft));
	set(ha2right,'Color',[1 1 1]*0.8);
	ch = get(ha2right,'Children');
	set(ch,'String',untex(letterright));
	drawnow
	status = 0;
      end
    end
  end
  
  
  subplot(ha1);
  inWindow= find(abs(mrk.pos-pp)/mrk.fs<MARKER_VIEW);
  nMark= length(inWindow);
  to = mrk.toe(inWindow);
  trig = cat(1,trigger{:,2});
  for ii= 1:nMark,
    [iii,jjj] = find(to(ii)==trig);
    if ~isempty(iii)
      x= (mrk.pos(inWindow(ii))-pp)*1000/mrk.fs;
      set(ht_marker(ii), 'position', [x 0 0], 'string',trigger{iii,1}, ...
			'color',colors(iii,:));
    end
  end
  set(ht_marker(nMark+1:end), 'position',[NaN 0 0]);
  drawnow;
end










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lettertree,pos] = create_lettertree(letters,probs,pos,parent,lettertree);


lettertree(pos).parent = parent;

probs = probs/sum(probs);

[dum,ind] = min(abs(cumsum(probs)-0.5));

lettertree(pos).left = {letters{1:ind}};
lettertree(pos).right = {letters{ind+1:end}};

pl = pos;

if length(lettertree(pl).left)==1
  lettertree(pl).leftchild = [];
else
  lettertree(pl).leftchild = pos+1;
  [lettertree,pos] = create_lettertree(lettertree(pl).left, ...
				       probs(1:ind),pos+1,pl,lettertree);
end

if length(lettertree(pl).right)==1
  lettertree(pl).rightchild = [];
else
  lettertree(pl).rightchild = pos+1;
  [lettertree,pos] = create_lettertree(lettertree(pl).right, ...
				       probs(ind+1:end),pos+1,pl,lettertree);
end


  