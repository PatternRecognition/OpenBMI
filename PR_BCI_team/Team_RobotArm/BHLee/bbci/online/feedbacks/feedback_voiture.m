function opt = feedback_voiture(fig, opt, lenk, brems);

persistent voi state hv hf hr hd ht hw hhm target waiting xLim yLim
persistent counter stopwatch running_time log_file fid

global lost_packages

if nargin<4,
  brems= 0;
end

if ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt= set_defaults(opt, ...
                    'acceleration', 3, ...
                    'max_speed', 10, ...
                    'max_angle', 45, ...
                    'speed_control', 1, ...
                    'carlength', 0.15, ...
                    'carwidth', 0.03, ...
                    'show_direction', 'off', ...
                    'show_front', 'on', ...
                    'show_rear', 'on', ...
                    'show_stopwatch', 'on', ...
                    'show_points', 'on', ...
                    'background', 0.9*[1 1 1], ...
                    'bumper_width', 10, ...
                    'car_color', [0 0 0], ...
                    'front_color', [1 1 0], ...
                    'rear_color', [1 0 0], ...
                    'direction_color', [0 0 1], ...
                    'target_color', [0.5 1 0.5], ...
                    'hit_color', [0 0.9 0], ...
                    'obstacle_color', [1 0.5 0.5], ...
                    'miss_color', [1 0 0], ...
                    'size_target', 0.25, ...
                    'size_obstacles', 0.1, ...
                    'obstacles', 2, ...
                    'time_after_hit', 500, ...
                    'time_before_next', 500, ...
                    'bounce', 0.1, ...
                    'torus', 1, ...
                    'car_type', 'line', ...
                    'log', 1, ...
                    'changed', 0, ...
                    'parPort',1,...
                    'position', get(fig,'position'));

  opt.reset = 0;

  %% prepare graphic objects for targets and obstacles
  target= zeros(opt.obstacles+1, 4);

  
  [handle,voi,xLim,yLim] = feedback_voiture_init(fig,opt);
  do_set('init',handle,'voiture',opt);
  do_set(200);

  hw=1;hhm=2;hv=3;hd=4;hf=5;hr=6;fig=7;gc=8;ht=9:length(handle);
  %% prepare bookkeeping data
  counter= zeros(1, 2);
  stopwatch= zeros(1, 2);
  state= 1;
end

if opt.changed==1 
  opt.changed= 0;
end

switch(state),
 case 1, %% choose target
  sz= opt.size_target;  %% size of the target
  dd= sz;  %% distance to border
  tt= [xLim(1)+dd+rand*(diff(xLim)-2*dd); yLim(1)+dd+rand*(diff(yLim)-2*dd)];
  target(1,:)= tt([1 2 1 2])' + [-1 -1 1 1]*sz/2;
  do_set(ht(1), 'xData',target(1,[1 3 3 1]), 'yData',target(1,[2 2 4 4]));
  sz= opt.size_obstacles;
  for ii= 1:opt.obstacles,
    tt= [xLim(1)+dd+rand*(diff(xLim)-2*dd); yLim(1)+dd+rand*(diff(yLim)-2*dd)];
    target(1+ii,:)= tt([1 2 1 2])' + [-1 -1 1 1]*sz/2;
    rect= target(1+ii,[1 2 1 2]) + [0 0 target(1+ii,[3 4])];
    do_set(ht(1+ii), 'xData',target(1+ii,[1 3 3 1]), ...
                  'yData',target(1+ii,[2 2 4 4]));
  end
  do_set(ht(1), 'faceColor',opt.target_color);
  do_set(ht(2:end), 'faceColor',opt.obstacle_color);
  do_set(ht, 'visible','on');
  running_time= 0;
  state= 2;
 case 2, %% wait for target hit
  mm= mean([voi.fw voi.rw],2);
  touch= pointinrect(mm, target);
  if ~isempty(touch),
    hit_or_miss= min([touch, 2]);
    counter(hit_or_miss)= counter(hit_or_miss)+1;
    do_set(hhm, 'string', sprintf('%d:%d', counter));
    stopwatch(hit_or_miss)= stopwatch(hit_or_miss) + running_time;
    do_set(9+hit_or_miss);  %% 10:hit, 11:miss
    if touch==1,
      do_set(ht(touch), 'faceColor',opt.hit_color);
    else
      do_set(ht(touch), 'faceColor',opt.miss_color);
    end
    do_set(setdiff(ht, ht(touch)), 'visible','off');
    waiting= opt.time_after_hit;
    state= 3;
  else
    running_time= running_time + 40*(1+lost_packages);
    mins= floor(running_time/60000);
    secs= floor((running_time-60000*mins)/1000);
    do_set(hw, 'string', sprintf('%d:%02d', mins, secs));
  end
 case 3, %% wait after target hit
  waiting= waiting - 40;
  if waiting<0, 
    do_set(ht, 'visible','off');
    waiting= opt.time_before_next;
    state= 4;
  end
 case 4,
  waiting= waiting - 40;
  if waiting<0,
    state= 1;
  end
end

    
%% update acceleration, speed, and steering angle (alpha)
switch(opt.speed_control),
 case 1, %% brake only
  voi.acc= opt.acceleration * (1 + min(0, 2*brems));
 case 2, %% accelerate only
  voi.acc= opt.acceleration * (-1 + max(0, 2*brems));
 case 3, %% control both
  voi.acc= opt.acceleration * brems;
end
voi.speed= voi.speed + voi.acc;
voi.speed= min(voi.speed, opt.max_speed);
voi.speed= max(voi.speed, 0);
voi.alpha= lenk*opt.max_angle/180*pi;

%% calculate direction vector (rv)
vv= voi.fw-voi.rw;
gamma= atan2(vv(1), vv(2));
rv= [sin(gamma+voi.alpha); cos(gamma+voi.alpha)] * voi.speed/1000;
%% move front wheels in new direction (rv)
voi.fw= voi.fw + rv;
%% move rear wheels in old direction (vv)
vn= vv'*vv;
ww= vv + rv;
p= (vv'*ww)/vn;
dq= (opt.carlength^2-(ww'*ww))/vn + p^2;
t1= p+sqrt(dq);
t2= p-sqrt(dq);
t= min([t1 t2]);
voi.rw= voi.rw + t * vv;

if opt.torus,
  tt= [0; 0];
  if voi.fw(1)<xLim(1),
    tt(1)= diff(xLim);
  elseif voi.fw(1)>xLim(2),
    tt(1)= -diff(xLim);
  end
  if voi.fw(2)<yLim(1),
    tt(2)= diff(yLim);
  elseif voi.fw(2)>yLim(2),
    tt(2)= -diff(yLim);
  end
  voi.fw= voi.fw + tt;
  voi.rw= voi.rw + tt;
else  
%% bouncing
  if voi.fw(1)<xLim(1) | voi.fw(1)>xLim(2),
    if voi.fw(1)<xLim(1),
      voi.fw(1)= xLim(1);
    else
      voi.fw(1)= xLim(2);
    end
    bounce= -sign(voi.fw(1)) * opt.bounce;
    voi.fw= voi.fw + [bounce; 0];
    voi.rw= voi.rw + [bounce; 0];
    voi.speed= 0;  
  end
  if voi.fw(2)<yLim(1) | voi.fw(2)>yLim(2),
    if voi.fw(2)<yLim(1),
      voi.fw(2)= yLim(1);
    else
      voi.fw(2)= yLim(2);
    end
    bounce= -sign(voi.fw(2)) * opt.bounce;
    voi.fw= voi.fw + [0; bounce];
    voi.rw= voi.rw + [0; bounce];
    voi.speed= 0;  
  end
end

%% calculate the corners of the car and draw it
vv= voi.fw-voi.rw;
vvo= [-vv(2); vv(1)]/sqrt(vv'*vv)*opt.carwidth;
fl= voi.fw + vvo;
fr= voi.fw - vvo;
rl= voi.rw + vvo;
rr= voi.rw - vvo;
vv= voi.fw - voi.rw;
bv= vv/sqrt(vv'*vv)*0.1*opt.carlength;
bfl= voi.fw + bv + vvo;
bfr= voi.fw + bv - vvo;
brl= voi.rw - bv + vvo;
brr= voi.rw - bv - vvo;

switch(opt.car_type),
 case 'patch',
  do_set(hv, 'xData', [fl(1) fr(1) rr(1) rl(1)], ...
          'yData', [fl(2) fr(2) rr(2) rl(2)]);
 case 'line',
  do_set(hv, 'xData', [voi.fw(1) voi.rw(1)], 'yData', [voi.fw(2) voi.rw(2)]);
 case 'marker', 
  do_set(hv, 'xData', mean([voi.fw(1) voi.rw(1)]), ...
          'yData', mean([voi.fw(2) voi.rw(2)]));
end

do_set(hf, 'xData', [bfl(1) bfr(1)], 'yData',[bfl(2) bfr(2)]);
do_set(hr, 'xData', [brl(1) brr(1)], 'yData',[brl(2) brr(2)]);
pp= [sin(voi.alpha+gamma) cos(voi.alpha+gamma)]*opt.carlength/3;
do_set(hd, 'xData', [voi.fw(1) voi.fw(1)+pp(1)], ...
        'yData', [voi.fw(2) voi.fw(2)+pp(2)]);
do_set('+');


