function fb_opt = feedback_dasher(fig, fb_opt, x, y);

persistent  xpos ypos angle rot_dir rot_active radius

if ~exist('y','var') | isempty(y)
  y = 0;
end


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
                       'fs', 25, ...
                       'log',0,...
                       'min_radius',.1,...
                       'x_range',[-1 1],...
                       'y_range',[-1 1],...
                       'rot_speed',1,...%how fast are we turning (constant speed)<positive scalar>
                       'rot_backforth',0,...%switch the direction of rotation ("Sumo style")<0,1>
                       'mapping_type',1,...%so far only 1 (Sumo) and 2 (pass-through) implemented.
                       'dasher_host','Brainamp',...
                       'dasher_port',20320,...
                       'rest_state',0)%show direction of gaze without acceleration; 0 means no rest.<positive scalar>.
  
  angle=0;  
  radius=fb_opt.min_radius;
  rot_dir = 1;
  rot_active=true;
  x_screen=0;
  y_screen=0;
  send_udp_dasher(fb_opt.dasher_host,fb_opt.dasher_port);
  do_set('init',[],'dasher',fb_opt);
  fb_opt.reset = 0;
end


switch fb_opt.mapping_type
    case 1
        % Sumo style: rotate and move.
        if x<0
            %rotation with only marginal movement required.
            if ~rot_active
                %the rotation now sets in after a movement/rest period.
                %maybe change direction.
                rot_active = true;
                if fb_opt.rot_backforth
                    rot_dir = -1*rot_dir;
                end
            end
            angle = angle +fb_opt.rot_speed*rot_dir; 
            radius = fb_opt.min_radius;
        elseif x<fb_opt.rest_state
            %no rotation, only marginal movement.
            rot_active=false;
            radius=fb_opt.min_radius;
        else   
            %no rotation, move forward.
            rot_active=false;
            radius = fb_opt.min_radius + (x-fb_opt.rest_state);
        end 
        % change to screen coordinates:
        x_screen = cos(angle*pi/180)*radius;
        x_screen = min(max(x_screen,fb_opt.x_range(1)),fb_opt.x_range(2));
        y_screen = sin(angle*pi/180)*radius;
        y_screen = min(max(y_screen,fb_opt.y_range(1)),fb_opt.y_range(2));
    case 2
        % Just forward the y position to the dasher
        x_screen = 0;
        y_screen = x;
end

send_udp_dasher(sprintf('x %d\n',x_screen));
send_udp_dasher(sprintf('y %d\n',y_screen));
%disp(sprintf('x %d y %d',x_screen,y_screen));
xpos = x; ypos=y;
