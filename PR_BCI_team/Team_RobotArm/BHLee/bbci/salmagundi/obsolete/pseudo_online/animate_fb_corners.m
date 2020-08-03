function fb_opt= animate_fb_corners(fb_opt, ptr, dscr_out,varargin)
% fb_opt= animate_fb_corners(fb_opt, ptr, dscr_out,varargin)
% 
% online feedback animation for three-class classification.
% plots three scalp-Patterns in three corners of the screen and 
% moves the cursor into the direction of the picture associated with the
% classifier with the highest output.
% IN: fb_opt - possible fields: 
%        .classOrder - array indicating the permutation of 
%                      classes (default [1 2 3], meaning left right bottom)
%        .mnt        - electrode montage (see plotScalpPattern)
%        .w          - nx3-matrix (see plotScalpPattern)
%        .rad        - radius of the corners, between 0 and .5
%        .crossSize  - size of the fixing cross.
%        .subtractMode - 1 - the third largest classifier is subtracted from the others
%                    - 2 - the second largest classifier is subtracted from the first.
%        .integrate  - number of passed values to calculate the average over.
%        .scale      - scaling factor for classification values; default 1.
%        .gameMode   - 1 - free gaming mode.
%                    - 2 - one of the circles is randomly chosen as a goal. 
%        .writeUdp   - use ppTrigger to write response triggers to udp. default false.
%        .cross_activation_time - how many times this function must be called 
%                      before activation of the cursor.
%        .cross_display_time - how many times this function must be called before
%                      erasing the changed colors of the circles
%        .pause      - (default false) - waits for keypress to continue

% kraulem 03/11/28

% static variables
persistent dscr_out_ma circles head_plots cross_plot count w h activeCircle center_y;
persistent cursor_active hitCircle gameType


if ~exist('fb_opt','var'), fb_opt=[]; end
if ~isfield(fb_opt,'classOrder'), fb_opt.classOrder = [1 2 3]; end
if ~isfield(fb_opt,'rad'), fb_opt.rad = .4; end
if ~isfield(fb_opt,'crossSize'), fb_opt.crossSize = .1; end
if ~isfield(fb_opt,'subtractMode'), fb_opt.subtractMode = 1; end
if ~isfield(fb_opt,'integrate'), fb_opt.integrate = 1; end
if ~isfield(fb_opt,'scale'), fb_opt.scale = 1; end
if ~isfield(fb_opt,'gameMode'), fb_opt.gameMode = 1; end
if ~isfield(fb_opt,'threshold'), fb_opt.threshold = 0; end
if ~isfield(fb_opt,'writeUdp'), fb_opt.writeUdp = false; end 
if ~isfield(fb_opt,'cross_activation_time'), fb_opt.cross_activation_time = 5; end 
if ~isfield(fb_opt,'cross_display_time'), fb_opt.cross_display_time = 10; end 
if ~isfield(fb_opt,'pause'), fb_opt.pause = false; end 

%if ~exist('gameType','var'), gameType = fb_opt.gameMode; end

if isequal(ptr, 'init')
  % start up
  clf;
  set(gcf,'MenuBar','none', 'color',0.4*[1 1 1]);
  set(gcf,'DoubleBuffer','on');
  scalp_width = .25;
  scalp_height = .25;
  mi = min(fb_opt.w(:));
  ma = max(fb_opt.w(:));
  head_plots(1) = axes('position', [0 (1-scalp_height) scalp_width scalp_height]);
  plotScalpPattern(fb_opt.mnt, fb_opt.w(:,fb_opt.classOrder(1)), ...
                   struct('scalePos', 'none','colAx',[mi ma]) );
               
  head_plots(2) = axes('position', [(1-scalp_width) (1-scalp_height) ...
                    scalp_width scalp_height]);
  plotScalpPattern(fb_opt.mnt, fb_opt.w(:,fb_opt.classOrder(2)), ...
                   struct('scalePos', 'none','colAx',[mi ma]) );

  head_plots(3) = axes('position', [(1-scalp_width)/2 0 scalp_width scalp_height]);
  plotScalpPattern(fb_opt.mnt, fb_opt.w(:,fb_opt.classOrder(3)), ...
                   struct('scalePos', 'none','colAx',[mi ma]) );

  cross_plot = axes('position', [0 0 1 1]);
  
  % plot triangle and quartercircles
  t = [0:.1:(pi/2), (pi/2)];
  
  ar = get(gcf,'Position');
  h = ar(4);
  w = ar(3);
  
  x1 = h/w*[fb_opt.rad*cos(-t)];
  y1 = [1-fb_opt.rad*sin(t)];
  x2 = [1-fb_opt.rad*cos(-t)*h/w];
  y2 = [1-fb_opt.rad*sin(t)];
  x3 = [.5-fb_opt.rad/sqrt(2)*cos(t)*h/w, .5+  fb_opt.rad/sqrt(2)*cos(t(end:-1:1))*h/w];
  y3 = [fb_opt.rad/sqrt(2)*sin(t), fb_opt.rad/sqrt(2)*sin(t(end:-1:1))];
  hold on;
  center_y = 5/8;% (.25*w^2 + h^2)/(2*h*h)
  line([.5 .5], [center_y-fb_opt.crossSize center_y],'Color','k');
  line([.5 .5-.5*fb_opt.crossSize/sqrt(.25+(1-center_y)^2)],...
       [center_y center_y+fb_opt.crossSize*(1-center_y)/sqrt(.25+(1-center_y)^2)],'Color','k');
  line([.5 .5+.5*fb_opt.crossSize/sqrt(.25+(1-center_y)^2)],...
       [center_y center_y+fb_opt.crossSize*(1-center_y)/sqrt(.25+(1-center_y)^2)],'Color','k');
  
  
  circles = [plot(x1,y1) plot(x2,y2) plot(x3,y3)];

  set(circles(1),'EraseMode','xor','LineWidth',8,'Color','k');
  set(circles(2),'EraseMode','xor','LineWidth',8,'Color','k');
  set(circles(3),'EraseMode','xor','LineWidth',8,'Color','k');
 
  cross_plot = plot([.5],[.75],'+');
  axis off
  hitCircle = [];
  if fb_opt.gameMode == 2
    activeCircle =[];
      %activeCircle = ceil(rand*3);
    %set(circles(activeCircle),'Color','g');
    %if fb_opt.writeUdp
    %  ppTrigger(fb_opt.classOrder(activeCircle));
    %end
    count = 10*fb_opt.cross_activation_time;
    cursor_active = false;
    set(cross_plot,'EraseMode','xor','MarkerSize',60,'LineWidth',7,'Color','r');
  else
    activeCircle = [];
    cursor_active = true;
    count = [];
    set(cross_plot,'EraseMode','xor','MarkerSize',60,'LineWidth',7,'Color','b');
  end
  gameType = fb_opt.gameMode;
else
  % refresh the inner picture with current classifier outputs.
%  if gameType~=fb_opt.gameMode
%    animate_fb_corners(fb_opt,'init', dscr_out,varargin);
%    gameType = fb_opt.gameMode;
%  end
  fb_opt.gameMode = gameType;
  p0= max(1, ptr-fb_opt.integrate+1);% mean over the last classifier outputs
  dscr_out_ma(:,ptr)= mean(dscr_out(:,p0:ptr),2);
  mv_vec = [[-.5 (1-center_y)]; [.5 (1-center_y)]; [0 -center_y]];
  class_out = dscr_out_ma(:,ptr);
  class_out = class_out(fb_opt.classOrder);
  mv_vec = mv_vec(fb_opt.classOrder,:);
  
  % sort the classifier outputs 
  [dum, dum_ind] = sort(class_out);
  if fb_opt.subtractMode==1
    class_out(dum_ind(3)) = class_out(dum_ind(3))- class_out(dum_ind(1));
    class_out(dum_ind(2)) = class_out(dum_ind(2))- class_out(dum_ind(1));
    class_out(dum_ind(1)) = 0;
    class_out =max(0,class_out-fb_opt.threshold);
    X_coord = fb_opt.scale*class_out(dum_ind(3))*mv_vec(dum_ind(3),:) ...
              + fb_opt.scale*class_out(dum_ind(2))*mv_vec(dum_ind(2),:) + [.5 center_y];
    X_coord = max(min(X_coord,1),0);% data is scaled to [0 1].
  elseif fb_opt.subtractMode==2
    class_out(dum_ind(3)) = class_out(dum_ind(3))- class_out(dum_ind(2));
    class_out(dum_ind(2)) = 0;
    class_out(dum_ind(1)) = 0;
    class_out =max(0,class_out-fb_opt.threshold);
    X_coord = fb_opt.scale*class_out(dum_ind(3))*mv_vec(dum_ind(3),:) ...
              + [.5 center_y];
    X_coord = max(min(X_coord,1),0);% data is scaled to [0 1].
  end
  
  % set Cross to X_coord
  set(cross_plot,'XData',X_coord(1),'YData',X_coord(2));
  
  % check if X_coord is within one of the circles
  if cursor_active
    if (X_coord(1)*w/h)^2+(1-X_coord(2))^2<fb_opt.rad^2
      % left upper circle
      %set(circles(1),'Color','g');
      hitCircle = 1;
    elseif ((1-X_coord(1))*w/h)^2 + (1-X_coord(2))^2 < fb_opt.rad^2
      % right upper circle
      hitCircle = 2;
    elseif ((.5-X_coord(1))*w/h)^2 + (X_coord(2))^2 < fb_opt.rad^2/2
      % lower circle
      hitCircle = 3;
    end
  end
  
  if fb_opt.gameMode == 1
    % draw colors for circles
    if cursor_active&isempty(count)&~isempty(hitCircle)
      % have freshly hit a new circle - start countdown
      count = fb_opt.cross_display_time;
      cursor_active = false;
      set(cross_plot,'Color','r');
      if fb_opt.writeUdp
        ppTrigger(fb_opt.classOrder(hitCircle)+ 10);
      end
      set(circles(hitCircle),'Color',[0 1 0]);  
    end
      
    if ~isempty(count)&~isempty(hitCircle)
      if count>0
        count = count-1;
      else
        %counter has run out. reset.
        set(circles(hitCircle),'Color','k');
        hitCircle = [];
        count = fb_opt.cross_activation_time;
      end
    elseif ~isempty(count)
      % after the hit: relax for  some time.
      if count>0
        count = count-1;
      else
        %counter has run out. reset.
        cursor_active = true;
        set(cross_plot,'Color','b');
        count = [];
      end      
    end
  elseif fb_opt.gameMode ==2
    if ~isempty(activeCircle)&~isempty(count)&~cursor_active
      % first phase: stimulus is shown; wait some time.
      if count>0
        count = count-1;
      else
        %counter has run out. wait for one of the circles to be hit.
        cursor_active = true;
        set(cross_plot,'Color','b');
        count = [];
      end            
    end
    
    if cursor_active&isempty(count)&~isempty(hitCircle)
      % have freshly hit a new circle - start countdown
      count = 2*fb_opt.cross_display_time;
      cursor_active = false;
      set(cross_plot,'Color','r');
      if hitCircle==activeCircle
        % success!
       set(circles(hitCircle),'Color',[0 1 0]);
      else
        % missed!
        set(circles(hitCircle),'Color','r');
        set(circles(activeCircle),'Color','k');
      end
      if fb_opt.writeUdp
        ppTrigger(fb_opt.classOrder(hitCircle)+10);
      end
      activeCircle = [];
    end
  
    if ~cursor_active&~isempty(count)&isempty(activeCircle)
      % after the hit.
      if count>fb_opt.cross_display_time|(count<fb_opt.cross_display_time&count>0)
        count = count-1;
      elseif count==fb_opt.cross_display_time
        % hide the hit circle color.
        set(circles(hitCircle),'Color','k');
        hitCircle = [];
        count = count-1;
      elseif count==0
        activeCircle = ceil(rand*3);
        set(circles(activeCircle),'Color',[0 0 1]);
        if fb_opt.writeUdp
          ppTrigger(fb_opt.classOrder(activeCircle));
        end
        count = fb_opt.cross_activation_time;
      end                  
    end
  end
    drawnow;
    if fb_opt.pause
      pause;
    end
end
return