function opt = feedback_flipper(fig, opt, cdata, varargin)
%FEEDBACK_FLIPPER - BBCI pinball control Feedback
%
%Synopsis:
% OPT = feedback_flipper(FIG, OPT, CTRL)
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
%propertyName(default)	Discription: 
% threshold:(0.75)      The threshold wenn the flippers will switch
% maxUptime(2000)       The time the one arm stayes maximal in the up position
% rebound_downtime(550)  The time the arm stayes in the down position during
%                       a rebound
% rebound_uptime(200)    The time the arm stayes in the up position during
%                       a rebound
% min_down_latency(400) The minimum time the arm has to rest after he was in upper position
% max_down_latency(1000)The maximum time the arm has to rest after he was in upper position
% rebound_thresh(500)   The threshold when the arm will do a rebound
% num_rebounds(1)       The number of rebounds the arm will do
% controlSwitch:(1)     With this switch you can gradually switch between
%                       rate control and position controll
%                       position control := 0
%                       rate control := 1
% rateScal:(1)          The scaling applied to the incoming cdata when
%                       using rate control
% posScal:(1)           The scaling applied to the incoming cdata when
%                       using position control
% sendParallel:(1)      Turn parallel port output on / off (for hardware pinball)
% sendUdp:(1)           Turn udp port sending on/off (for software pinball)
% sendIP                The ip adress to send when udp sending
% sendPort              The port to send to when udp sending
% log(1)                If logging of the parallel port and gui commands is
%                       enabled

% Author(s): Max Sagebaum, Jun-2007

% two stuctures for represeting the state of the left and right flipper
persistent leftFlipper rightFlipper;
persistent internalCData;					%the intenal classifier value
 
	% values for the display
  INCOMMING_DATA = 1;
  INCOMMINT_DATA_STATUS = 2;
  INTERNAL_DATA = 3;
  INTERNAL_DATA_STATUS = 4;
  LEFT_FLIPPER = 5;
  RIGHT_FLIPPER = 6;
  
	if ~isfield(opt,'reset')
 		opt.reset = 1;
	end
	
	if (opt.reset == 1)
  		opt.reset = 0;
  		opt= set_defaults(opt, ...,
            'position',[0 0 800 600],...
            'threshold',0.75,...
            'maxUptime',2000,...
            'rebound_downtime',550,...
            'rebound_uptime',200,...
            'min_down_latency',400,...
            'max_down_latency',1000,...
            'rebound_thresh',500,...
            'num_rebounds',1,...
            'controlSwitch',1.0,...
            'rateScal',1,...
            'posScal',1,...
            'sendParallel',1,...
            'sendUdp',1,...
            'sendIP','127.0.0.1',...
            'sendPort',12489,...
            'log',1);
        opt.lastCall = javaMethod('nanoTime','java.lang.System');
        
        leftFlipper = createFlipper(1);
        rightFlipper = createFlipper(0);
        internalCData = 0;
        
        diss = feedback_flipper_init(fig,opt);
        [handles, H]= fb_handleStruct2Vector(diss);

        do_set('init',handles,'flipper',opt);
        
        % create an udp connection
        if( 1 == opt.sendUdp)
            send_data_udp
            send_data_udp(opt.sendIP,opt.sendPort);
            fprintf('Connect to %d at %s\n',opt.sendPort,opt.sendIP);

        end
  end
   
    % apply the incomming cdata to the internalCdata
    ratePart = internalCData * opt.controlSwitch + cdata * opt.rateScal * opt.controlSwitch;
    posPart = cdata * opt.posScal * (1 - opt.controlSwitch);
    internalCData =  ratePart + posPart;
    
    if(leftFlipper.state == 1)
        flipper_rotate(LEFT_FLIPPER,[0 0 1],-45,[45 45 0]);		% rotate back to the start position
    end
    if(rightFlipper.state == 1)
        flipper_rotate(RIGHT_FLIPPER,[0 0 1],45,[755 45 0]);		% rotate back to the start position
    end
    
    curTime = javaMethod('nanoTime','java.lang.System');
    timeLastCall = (curTime - opt.lastCall) * 10^-6;
    opt.lastCall = curTime;
      
    leftFlipper = updateFlipperState(leftFlipper,internalCData,opt,timeLastCall);
    rightFlipper = updateFlipperState(rightFlipper,internalCData,opt,timeLastCall);
    
	% clamp the internalCData to [-1, 1]
    if(internalCData < -1)
        internalCData = -1;
    else 
        if(internalCData > 1)
            internalCData = 1;
        end
    end 
    
	% update the figure
	setScalX(INCOMMING_DATA, INCOMMINT_DATA_STATUS,cdata);
	setScalX(INTERNAL_DATA,INTERNAL_DATA_STATUS,internalCData);
    if(leftFlipper.state == 1)
        flipper_rotate(LEFT_FLIPPER,[0 0 1],45,[45 45 0]);		% rotate to up position
    end
    if(rightFlipper.state == 1)
        flipper_rotate(RIGHT_FLIPPER,[0 0 1],-45,[755 45 0]);		% rotate to up position
    end
    
    if( 1 == opt.sendParallel)
        % write to the parallel port the states of the left and right flippers
        ppValue = 0;
        if(leftFlipper.state == 1)
            ppValue = ppValue + 4;
        end
        if(rightFlipper.state == 1)
            ppValue = ppValue + 1;
        end
        
        do_set('trigger',ppValue,0);
        
    end
    if( 1 == opt.sendUdp )
        package = zeros(1,5);

          package(2) = package(2)+1;
          package(3) = 30;
          if(leftFlipper.state == 1)
              package(5) = 1.0;
          else
              if(rightFlipper.state == 1)
                  package(5) = -1.0;
              else
                  package(5) = 0.0;
              end
          end
          disp(package(5));
          send_data_udp(package);
    end
    
    do_set('+');
%    disp(leftFlipper);
%    disp(rightFlipper);
    
    function flipper = createFlipper(isLeft)
       flipper = struct('state',0,...
           'timeInState',0,...
           'reboundsLeft',0,...
           'downLatencyLeft',0,...
           'reboundActive',0,...
           'isLeft',isLeft);
    end

    function flipper = updateFlipperState(flipper, value,opt,timeLastCall)
        flipper.timeInState = flipper.timeInState + timeLastCall;
        
        % check if we have to rebound
        if(flipper.reboundActive == 1)
            if(flipper.state == 1 &&  flipper.timeInState >= opt.rebound_uptime)
                flipper.state = 0;
                flipper.timeInState = 0;

                if(flipper.reboundsLeft == 0)
                    % this was the last rebound
                    flipper.reboundActive = 0;
                end
            end

            if(flipper.state == 0 &&  flipper.timeInState >= opt.rebound_downtime)
                flipper.state = 1;
                flipper.timeInState = 0;

                flipper.reboundsLeft = flipper.reboundsLeft - 1;
            end
        elseif(flipper.downLatencyLeft ~= 0)
            % the flipper has to rest in idle state 
            flipper.downLatencyLeft = flipper.downLatencyLeft - timeLastCall;
            if(flipper.downLatencyLeft < 0) 
                flipper.downLatencyLeft = 0;
            end
        else
        % the flipper is free
            overThreshold = 0;
            if(flipper.isLeft == 1)
                if(value < -opt.threshold)
                    overThreshold = 1;
                end
            else
                if(value > opt.threshold)
                    overThreshold = 1;
                end
            end

            if(flipper.state == 1 && overThreshold == 1)
                %only check if maxUptime was reached
                if(opt.maxUptime <= flipper.timeInState)
                    flipper.reboundsLeft = getReboundCount(flipper.timeInState,opt);
                    flipper.reboundActive = 1;
                    flipper.downLatencyLeft = getDownLatency(flipper.timeInState,opt);
                end
            elseif (flipper.state == 1 && overThreshold == 0)
                flipper.reboundsLeft = getReboundCount(flipper.timeInState,opt);
                flipper.reboundActive = 1;
                flipper.downLatencyLeft = getDownLatency(flipper.timeInState,opt);
            elseif (flipper.state == 0 && overThreshold == 1)
                flipper.state = 1;
                flipper.timeInState = 0;
            else
                % do nothing
            end
        end
    end

    function latency = getDownLatency(timeInState,opt)
        latency = opt.min_down_latency + (timeInState / opt.maxUptime) * (opt.max_down_latency - opt.min_down_latency);
    end 

    function rebound = getReboundCount(timeInState,opt)
        rebound = 0;
        if(timeInState >= opt.rebound_thresh)
            rebound = opt.num_rebounds;
        end
    end

	% sets the width of one status bar
	function setBarX(bar,xStart,xEnd)
        do_set(bar,'XData', [xStart xStart;xEnd xEnd;xStart xEnd]);        
  end
    % sets the width of one status bar according to the size of the backbar and scaling
	% scaling shuld be from -1 to 1
	function setScalX(backBar,frontBar,scal)
        xData = do_set('get',backBar,'XData');
        xStart = xData(1,1) + 5;
        xEnd = xData(2,1) - 5;
        xCenter = (xStart + xEnd) / 2;
        xWidth = (xEnd - xStart) / 2;
        xPos = xWidth * scal + xCenter;
        
        if(xPos < xCenter)
             setBarX(frontBar,xPos,xCenter);
        else
            setBarX(frontBar,xCenter,xPos);
        end
  end

   function flipper_rotate(h,azel,alpha,origin)
  %ROTATE Rotate objects about specified origin and direction.
  %   ROTATE(H,[THETA PHI],ALPHA) rotates the objects with handles H
  %   through angle ALPHA about an axis described by the 2-element
  %   direction vector [THETA PHI] (spherical coordinates).  
  %   All the angles are in degrees.  The handles in H must be children
  %   of the same axes.
  %
  %   THETA is the angle in the xy plane counterclockwise from the
  %   positive x axis.  PHI is the elevation of the direction vector
  %   from the xy plane (see also SPH2CART).  Positive ALPHA is defined
  %   as the righthand-rule angle about the direction vector as it
  %   extends from the origin.
  %
  %   ROTATE(H,[X Y Z],ALPHA) rotates the objects about the direction
  %   vector [X Y Z] (Cartesian coordinates). The direction vector
  %   is the vector from the center of the plot box to (X,Y,Z).
  %
  %   ROTATE(...,ORIGIN) uses the point ORIGIN = [x0,y0,y0] as
  %   the center of rotation instead of the center of the plot box.
  %
  %   See also SPH2CART, CART2SPH.

  %   Copyright 1984-2005 The MathWorks, Inc. 

  % Determine the default origin (center of plot box).
  if nargin < 4
    ax = ancestor(h(1),'axes');
    if isempty(ax) || ax==0,
      error(id('InvalidHandle'),'H must contain axes children only.');
    end
    origin = sum([do_set('get',ax,'xlim')' do_set('get',ax,'ylim')' do_set('get',ax,'zlim')'])/2;
  end

  % find unit vector for axis of rotation
  if numel(azel) == 2 % theta, phi
      theta = pi*azel(1)/180;
      phi = pi*azel(2)/180;
      u = [cos(phi)*cos(theta); cos(phi)*sin(theta); sin(phi)];
  elseif numel(azel) == 3 % direction vector
      u = azel(:)/norm(azel);
  end

  alph = alpha*pi/180;
  cosa = cos(alph);
  sina = sin(alph);
  vera = 1 - cosa;
  x = u(1);
  y = u(2);
  z = u(3);
  rot = [cosa+x^2*vera x*y*vera-z*sina x*z*vera+y*sina; ...
         x*y*vera+z*sina cosa+y^2*vera y*z*vera-x*sina; ...
         x*z*vera-y*sina y*z*vera+x*sina cosa+z^2*vera]';

  for i=1:numel(h),
    t = do_set('get',h(i),'type');
    skip = 0;
    if strcmp(t,'surface') || strcmp(t,'line') || strcmp(t,'patch')

      % If patch, rotate vertices  
      if strcmp(t,'patch')
         verts = do_set('get',h(i),'Vertices');
         x = verts(:,1); y = verts(:,2); 
         if size(verts,2)>2
            z = verts(:,3);
         else
            z = [];
         end

      % If surface or line, rotate {x,y,z}data   
      else
         x = do_set('get',h(i),'xdata');
         y = do_set('get',h(i),'ydata');
         z = do_set('get',h(i),'zdata');
      end

      if isempty(z)
         z = -origin(3)*ones(size(y));
      end
      [m,n] = size(z);
      if numel(x) < m*n
        [x,y] = meshgrid(x,y);
      end
    elseif strcmp(t,'text')
      p = do_set('get',h(i),'position');
      x = p(1); y = p(2); z = p(3);
    elseif strcmp(t,'image')
      x = do_set('get',h(i),'xdata');
      y = do_set('get',h(i),'ydata');
      z = zeros(size(x));
    else
      skip = 1;
    end

    if ~skip,
      [m,n] = size(x);
      newxyz = [x(:)-origin(1), y(:)-origin(2), z(:)-origin(3)];
      newxyz = newxyz*rot;
      newx = origin(1) + reshape(newxyz(:,1),m,n);
      newy = origin(2) + reshape(newxyz(:,2),m,n);
      newz = origin(3) + reshape(newxyz(:,3),m,n);

      if strcmp(t,'surface') || strcmp(t,'line')
        do_set(h(i),'xdata',newx,'ydata',newy,'zdata',newz);
      elseif strcmp(t,'patch')
        do_set(h(i),'Vertices',[newx,newy,newz]); 
      elseif strcmp(t,'text')
        do_set(h(i),'position',[newx newy newz])
      elseif strcmp(t,'image')
        do_set(h(i),'xdata',newx,'ydata',newy)
      end
    end
  end
   end
end
