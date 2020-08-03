function [fb_opt,handle] = feedback_reactive(fig,fb_opt,class);

persistent status mess score classes fix_Point paus nClasses sequence pos trig relax


fb_opt = set_defaults(fb_opt,'reset',1,...
                             'fixPoint',1,...
                             'fixPointWidth',0.05,...
                             'fixPointColor',[0 0 1],...
                             'font_size',0.2,...
                             'block_size',0.3,...
                             'countdown',20,...
                             'block_line',2,...
                             'events',40,...
                             'font_color',[0 0 0],...
                             'pause_inbetween',[0.5 0.5],...
                             'score_size',0.1,...
                             'score_color',[0 0 0],...
                             'visualisation','letter',...
                             'show_result',0.5,...
                             'show_score',1,...
                             'stimulus','LRFX',...
                             'time_out',inf,...
                             'correct_color',[0 1 0],...
                             'wrong_color',[1 0 0],...
                             'block_color',[1 0 0],...
                             'changed',1,...
                             'fixPoint_params',{'FaceAlpha',0.5});


fb_opt = set_defaults(fb_opt,'time_out_relax',min(5,fb_opt.time_out));


if fb_opt.reset
  handle = feedback_reactive_init(fig,fb_opt);
  mess = 1;
  score = 2;
  fix_point = 3;
  classes = handle(4:end);
  do_set('init',handle,'reactive',fb_opt);
  status = 0;
  fb_opt.changed = 1;
  nClasses = length(fb_opt.stimulus);
  if length(fb_opt.events)==1
    fb_opt.events = ones(1,nClasses)*fb_opt.events;
  end
  sequence = zeros(1,sum(fb_opt.events));
  sequence(cumsum([1,fb_opt.events(1:end-1)])) = 1;
  sequence = cumsum(sequence);
  sequence = sequence(randperm(sum(fb_opt.events)));
  pos = 1;
end



if fb_opt.changed==1
  if strcmp(fb_opt.status,'pause') 
    do_set(mess,'String','pause','Visible','on');
    reactive_task(fb_opt,'relax',classes);
    if status<10 & status>0
      status = status+10;
    end
  elseif strcmp(fb_opt.status,'stop') 
    do_set(mess,'String','stop','Visible','on');
    reactive_task(fb_opt,'relax',classes);
    status = 0;
  else 
    if status==0
      status = 1;
    elseif status>10
      status = status-10;
    end
  end
end


if status>10 & status==0
  do_set('+');
  return;
end

if status==1
  % prepare countdown
  tic;
  do_set(mess,'String','Entspannen','Visible','on');
  status = 2;
end


if status==2  
  %COUNTDOWN
  tirest = fb_opt.countdown-toc;
  if tirest<5 & tirest>0
    do_set(mess,'String',int2str(ceil(tirest)));
  end
  if tirest<=0
    do_set(mess,'Visible','off');
    status = 3;
    paus = fb_opt.pause_inbetween(1)+rand(fb_opt.pause_inbetween(2));
    tic;
  end
end


if status==3
  % wait for order and finally order
  tirest = paus-toc;
  if tirest<=0
    trig = sequence(pos);
    pos = pos+1;
    do_set(trig);
    reactive_task(fb_opt,'order',classes,trig);
    status = 4;
    tic;
    if strcmp(sequence(trig),'X')
      paus = fb_opt.time_out_relax;
      relax = 1;
    else
      paus = fb_opt.time_out;
      relax = 0;
    end
  end
end

if status==4
  % wait for react and finally react
  tirest = paus-toc;
  if tirest<=0
    if relax
      do_set(mess,'Visible','on','String','000','Color',fb_opt.correct_color);
    else
      do_set(mess,'Visible','on','String','000','Color',fb_opt.wrong_color);
    end
    paus = fb_opt.show_result;
  end
  if class<0
    if abs(class)==trig
      do_set(mess,''); %% TO BE CONTINUED
      error('in construction');
    end
  end
end

if status==5
  % wait for next round
end

do_set('+');
