function fb_opt = feedback_bars(fig, fb_opt, varargin);

persistent no1 hax ha hb hl countdown active ordered hit gameMode
persistent hit_countdown blocktime start_countdown pointer 


marker = [];
nClasses = length(varargin);
if nClasses>9
  error('for more than 9 classes not implemented so far');
end
 
fb_opt = set_defaults(fb_opt,...
                      'classOrder',1:nClasses,...
                      'parPort',1,...
                      'position',get(fig, position),...
                      'event',1,...
                      'lighttime',8,...
                      'staytime',4,...
                      'yLim',[-1 1],...
                      'threshold',0,...
                      'gameMode',1,...
                      'hit_countdown',50,...
                      'blocktime',50,...
                      'countdown',0,...
                      'sequence','Z',...
                      'changed',0,...
                      'reset',1);
if gameMode~=fb_opt.gameMode
  %gameMode = fb_opt.gameMode;
  fb_opt.reset = 1;
  % fb_opt = feedback_bars(fig, fb_opt, varargin);
end
if ~isfield(fb_opt,'classes')
  fb_opt.classes = cell(1,nClasses);
end
  
dscr = [varargin{:}];


if fb_opt.reset==1
  gameMode = fb_opt.gameMode;
  hit_countdown = -1;
  blocktime = 0;
  start_countdown = fb_opt.countdown;
  pointer =0;
  handle = feedback_bars_init(fig,fb_opt);

  do_set('init',handle,'bars',fb_opt);
  do_set(200);
  
  ha = 1:nClasses;
  hb = nClasses+1:2*nClasses;
  hl = 2*nClasses+1:3*nClasses;
  hax = 3*nClasses+1:4*nClasses;
  fig = 4*nClasses+1;
  gc = 4*nClasses+2;
  active= 0;
  countdown= -1*ones(1,nClasses);
  hit_countdown = -1;
  fb_opt.reset = 0;
  ordered = 0;% ceil(rand*nClasses);
  fb_opt.changed = 0;
end

if fb_opt.changed ~=0;
  fb_opt.changed = 0;
end



if gameMode==1
  % this game doesn't order bars.
  above= ( dscr>fb_opt.threshold);
  if sum(above)>1, 
    if fb_opt.event 
      [dum,ma] = max(dscr);
      above(:)=0;
      above(ma) = 1;
    else
      above(:)= 0;
    end
  end
  for iii = 1:nClasses
    do_set(hb(iii), 'yData',[[1 1]*fb_opt.yLim(1) [1 1]*(dscr(iii))],...
               'faceColor',barColor(iii,:,2+above(iii)));
  end
  if sum(above)==1,
    actual= find(above);
    if actual~=active | countdown(actual)<0,
      if active>0,
        countdown(active)= 0;
      end
      active= actual;
      countdown(active)= fb_opt.lighttime;
      do_set(hax(active), 'color',barColor(active,:,2));
      do_set(30+active);	 
    else
      countdown(active)= max(fb_opt.staytime, countdown(active));
    end
  end
  
  turn_off= find(countdown==0);
  do_set(hax(turn_off), 'color',[1 1 1]);
  countdown= max(-1, countdown-1);
else 
  % game mode 2: order bars...
  for iii = 1:nClasses
    do_set(hb(iii), 'yData',[[1 1]*fb_opt.yLim(1) [1 1]*(dscr(iii))],...
                 'faceColor',barColor(iii,:,2+above(iii)));
  end
  if ordered == 0& start_countdown==0
    %nothing ordered yet.
    if hit_countdown ==-1
      %countdown has run out. New order.
      if ischar(fb_opt.sequence)
        if(strcmp(fb_opt.sequence,'Z'))
         ordered = ceil(rand*nClasses); 
        else
          pointer = pointer+1;
          if pointer>length(fb_opt.sequence)
            pointer = 1;
          end 
          ordered=fb_opt.sequence(pointer)-'0';
        end  
      end  
      do_set(ordered);
      do_set(hax(ordered),'color',[0 0 1]);
      blocktime = fb_opt.blocktime;
    else
      %only redraw, don't change colors.
    end
  end
  above = zeros(1,nClasses);
  if ordered~=0
    %find over-threshold values.
      above= ( dscr>fb_opt.threshold);
      if sum(above)>1, 
        if fb_opt.event 
          [dum,ma] = max(dscr);
          above(:)=0;
          above(ma) = 1;
        else
          above(:)= 0;
        end
      end
      if sum(above)==1,
        actual= find(above);
        if actual~=active | countdown(actual)<0,
          if active>0,
            countdown(active)= 0;
          end
          active= actual;
          countdown(active)= fb_opt.lighttime;
          %do_set(ha(active), 'color',barColor(active,:,2));
      else
          countdown(active)= max(fb_opt.staytime, countdown(active));
        end
        if blocktime==0
          if actual==ordered
           % hit
           do_set(10+ordered);
           do_set(hax(ordered),'color',[0 1 0]);
          else
            % miss
            do_set(20+actual);
           do_set(hax(actual),'color',[1 0 0]);
            do_set(hax(ordered),'color',[1 1 1]);
          end
          hit_countdown = fb_opt.hit_countdown;
          ordered =0;
        end   
    end
  end
  %redraw
  turn_off= find(countdown==0);
  do_set(hax(turn_off), 'color',[1 1 1]);
  countdown= max(-1, countdown-1);
  if hit_countdown==0
    for i =1:nClasses
      do_set(hax(i), 'color',[1 1 1]);
    end 
  end 
  hit_countdown = max(-1, hit_countdown-1);
  blocktime = max(0,blocktime-1);
  start_countdown = max(0,start_countdown-1);
end

do_set('+');
