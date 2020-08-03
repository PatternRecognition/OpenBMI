classdef JacoComm < matlab.System & matlab.system.mixin.Propagates ...
        & matlab.system.mixin.CustomIcon ...
        & matlab.system.mixin.internal.SampleTime
    % JacoComm Send commands and read sensor data from Jaco robot
    %  See testJacoComm.m for examples of how to use the class.
    %  Copyright 2017 The MathWorks, Inc.

    
    
    properties (Constant,Access = private)
        NumFingers = 3;
        CartParam = 6;
        MaxJoints = 7;
    end

    properties (Access = private)
        NumJoints = JacoComm.MaxJoints;
        JntPosCmd = zeros(JacoComm.MaxJoints,1);
        JntVelCmd = zeros(JacoComm.MaxJoints,1);
        JntTorqueCmd = zeros(JacoComm.MaxJoints,1);
        FingerPosCmd = zeros(JacoComm.NumFingers,1);
        CartPosCmd = zeros(JacoComm.CartParam,1);
        CartVelCmd = zeros(JacoComm.CartParam,1);
        OffsetCmd = zeros(4,1);
        ZoneCmd = zeros(18,1);
        
        %GoToHomeCmdPast - Previous value of input 
        GoToHomeCmdPastValue = false;
        
        % 
        CommandModePastValue = int32(-1);
    end

    % Public, non-tunable properties
    properties(Nontunable)
       ControlMode  = 'Position'; % Control Mode
       
    end
    
    % Public, non-tunable properties
    properties (Nontunable,Logical)
       CalibrateFingersAtStartup  = false; % Calibrate fingers at startup
       GoToHomeAtStartup = false;  % Send Go to Home command at startup
    end
    
    properties(Hidden, Nontunable)
       ControlModeSet  = matlab.system.StringSet({'Position','Velocity','Torque'}); 
    end
    
    properties(SetAccess = protected)
       IsConnected = false; 
    end

    properties(DiscreteState)

    end   


    % Pre-computed constants
    properties(Access = private)
    end
    
    properties(Dependent, SetAccess = protected)   
       JointPos;
       JointVel;
       JointTorque;
       JointTemp;
       FingerPos;
       FingerVel;
       FingerTorque;
       FingerTemp;
       EndEffectorPose;
       EndEffectorWrench;
       DOF;
       EndEffectorOffset;
       ProtectionZone;
       TrajectoryInfo;
    end

    methods
        % Constructor
        function obj = JacoComm(varargin)
            % Support name-value pair arguments when constructing object
            setProperties(obj,nargin,varargin{:});
        end
                

        function connect(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] = JacoMexInterface(double(MexFunctionIDs.OPEN_LIB),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else
                driverPath = 'C:\MATLAB\Demos\robotarm2\Source\Drivers\JACO2Communicator\JACO2SDK';
                coder.updateBuildInfo('addSourcePaths',driverPath);
                coder.updateBuildInfo('addSourceFiles','JacoSDKWrapper.cpp');
                coder.updateBuildInfo('addIncludePaths',driverPath);            
                stat = coder.ceval('openKinovaLibrary');
            end
            if ~stat
                error('Failed to load library and open API');
            end
            obj.IsConnected = true;
            % Setting the number of DOF when connecting to the arm
            obj = setNumJoints(obj);
        end
        
        function disconnect(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~]  = JacoMexInterface(double(MexFunctionIDs.CLOSE_LIB),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('closeKinovaLibrary');
            end
            if ~stat
                error('Failed to close library');
            end
            obj.IsConnected = false;
        end
        
        function y = setNumJoints(obj)
             value = getDOF(obj);
            if (value == 4 || value == 6 || value == 7)
                obj.NumJoints = value;
            else
                error('Numjoints must be 4, 6 or 7')
            end
            y = obj;
        end
                   
        function setPositionControlMode(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SET_POS_CTRL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('setPositionControlMode');
            end
            if ~stat
                error('Failed to change control mode');
            end
            obj.ControlMode = 'Position';
        end
        
        function setVelocityControlMode(obj)
            stat = false; %#ok<NASGU>
            % Velocity also uses position control mode
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SET_POS_CTRL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('setPositionControlMode');
            end
            if ~stat
                error('Failed to change control mode');
            end
            obj.ControlMode = 'Velocity';
        end
        
        function setTorqueControlMode(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SET_DIRECT_TORQUE_CTRL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('setTorqueControlMode');
            end
            if ~stat
                error('Failed to change control mode');
            end
            obj.ControlMode = 'Torque';
        end
        
        function goToHomePosition(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.MOVE_TO_HOME),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('moveToHomePosition');
            end
            if ~stat
                error('Failed to go to home position');
            end
        end
        
        function calibrateFingers(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.INIT_FINGERS),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('initializeFingers');
            end
            if ~stat
                error('Failed to calibrate fingers');
            end
        end
        
        
        function pos = getJointAndFingerPos(obj)
            %getJointAndFingerPos - Public property to query all the position vector 
            pos = zeros(obj.NumJoints+obj.NumFingers,1);
            if obj.IsConnected
                pos = obj.getPositions();
            else
                error('Not connected to robot');
            end
        end

        function vel = getJointAndFingerVel(obj)
            %getJointAndFingerVel - Public property to query all the velocity vector 
            vel = zeros(obj.NumJoints+obj.NumFingers,1);  %#ok<PREALL>
            if obj.IsConnected
                vel = obj.getVelocities();
            else
                error('Not connected to robot');    
            end
        end
        
        function torque = getJointAndFingerTorque(obj)
            %getJointAndFingerTorque - Public property to query all the torque vector 
            torque = zeros(obj.NumJoints+obj.NumFingers,1);  %#ok<PREALL>
            if obj.IsConnected
                torque = obj.getTorques();
            else
                error('Not connected to robot');    
            end
        end
        
        function temp = getJointAndFingerTemp(obj)
            %getJointAndFingerTemp - Public property to query all the temperature vector 
            temp = zeros(obj.NumJoints+obj.NumFingers,1);  %#ok<PREALL>
            if obj.IsConnected
                temp = obj.getTemperatures();
            else
                error('Not connected to robot');    
            end
        end
        
        function sendJointPositionCommand(obj,cmd)
            % function description
            
            % Validate input
            validateattributes(cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.NumJoints},'sendJointPositionCommand');
            
            % Call private function 
            obj.sendJointPositionCommandInternal(cmd);
        end
        
        function sendFingerPositionCommand(obj,cmd)
            % Validate input
            validateattributes(cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.NumFingers},'sendFingerPositionCommand');
            
            % Call private function 
            obj.sendFingerPositionCommandInternal(cmd);           
        end
        
        function sendJointAndFingerPositionCommand(obj,jCmd,fCmd)
            % Validate inputs
            validateattributes(jCmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.NumJoints},'sendJointAndFingerPositionCommand');
            validateattributes(fCmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.NumFingers},'sendJointAndFingerPositionCommand');            
            
            % Call private function 
            obj.sendJointAndFingerPositionCommandInternal(jCmd,fCmd);              
        end
        
        function sendJointVelocityCommand(obj,cmd)
            % Validate input
            validateattributes(cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.NumJoints},'sendJointVelocityCommand');
            
            % Call private function 
            obj.sendJointVelocityCommandInternal(cmd);
        end
        
        function sendJointTorqueCommand(obj,cmd)
            % Validate input
            validateattributes(cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.NumJoints},'sendJointTorqueCommand');            
            
            if strcmp(obj.ControlMode,'Torque')
            % Call private function 
                obj.sendJointTorqueCommandInternal(cmd);
            else
                error('Control mode not set in torque mode');
            end
        end
         
        function sendCartesianPositionCommand(obj,Cmd)
            % Validate inputs
            validateattributes(Cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.CartParam},'sendCartesianPositionCommand');
            
            % Call private function 
            obj.sendCartesianPositionCommandInternal(Cmd);              
        end
        
        function sendCartesianVelocityCommand(obj,Cmd)
            % Validate inputs
            validateattributes(Cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',obj.CartParam},'sendCartesianVelocityCommand');
            
            % Call private function 
            obj.sendCartesianVelocityCommandInternal(Cmd);              
        end
        
        function setEndEffectorOffset(obj,Cmd)
            % Validate inputs
            validateattributes(Cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',4},'setEndEffectorOffset');
            
            % Call private function 
            obj.setEndEffectorOffsetInternal(Cmd);              
        end
        
        function setProtectionZone(obj,Cmd)
            % Validate inputs
            validateattributes(Cmd, {'numeric'},...
                {'real', 'nonnan','nonempty','finite', 'column',...
                'nrows',18},'setProtectionZone');
            
            % Call private function 
            obj.setProtectionZoneInternal(Cmd);              
        end
        
        function runGravityCalibration(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.RUN_GRAVITY_CALIB),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('runGravityCalibration');
            end
            if ~stat
                error('Failed to run gravity calibration routine');
            end            
        end
        
        function StartForceControl(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.START_FORCE_CONTROL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('startForceControl');
            end
            if ~stat
                error('Failed to start force control');
            end
        end
        
        function StopForceControl(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.STOP_FORCE_CONTROL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('stopForceControl');
            end
            if ~stat
                error('Failed to stop force control');
            end
        end
        
        function EraseAllProtectionZones(obj)
            stat = false; %#ok<NASGU>
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.ERR_PROTECT_ZONES),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('eraseAllProtectionZones');
            end
            if ~stat
                error('Failed to erase protection zones');
            end
        end
                    
        % Get methods for public properties 
        function jointPos = get.JointPos(obj)
            if obj.IsConnected
                jointPos = obj.getJointPositions();
            else
                jointPos = [];
            end
        end
        
        function jointVel = get.JointVel(obj)
            if obj.IsConnected
                jointVel = obj.getJointVelocities();
            else
                jointVel = [];
            end
        end
        
        function jointTorque = get.JointTorque(obj)
            if obj.IsConnected
                jointTorque = obj.getJointTorques();
            else
                jointTorque = [];
            end
        end
        
        function jointTemp = get.JointTemp(obj)
            if obj.IsConnected
                jointTemp = obj.getJointTemperatures();
            else
                jointTemp = [];
            end
        end
        
        function fingerPos = get.FingerPos(obj)
            if obj.IsConnected
                fingerPos = obj.getFingerPositions();
            else
                fingerPos = [];
            end
        end
        
        function fingerVel = get.FingerVel(obj)
            if obj.IsConnected
                fingerVel= obj.getFingerVelocities();
            else
                fingerVel = [];
            end
        end  
        
        function fingerVel = get.FingerTorque(obj)
            if obj.IsConnected
                fingerVel= obj.getFingerTorques();
            else
                fingerVel = [];
            end
        end  
        
        function fingerTemp = get.FingerTemp(obj)
            if obj.IsConnected
                fingerTemp= obj.getFingerTemperatures();
            else
                fingerTemp = [];
            end
        end
        
        function pose = get.EndEffectorPose(obj)
            if obj.IsConnected
                pose= obj.getEndEffectorPose();
            else
                pose = [];
            end
        end
        
        function pose = get.EndEffectorWrench(obj)
            if obj.IsConnected
                pose= obj.getEndEffectorWrench();
            else
                pose = [];
            end
        end
        
        function DOF = get.DOF(obj)
            if obj.IsConnected
                DOF = obj.getDOF();
            else
                DOF = 0;
            end
        end
        
        function off = get.EndEffectorOffset(obj)
            if obj.IsConnected
                off = obj.getEndEffectorOffset;
            else
                off = [];
            end
        end
        
        function zone = get.ProtectionZone(obj)
            if obj.IsConnected
                zone = obj.getProtectionZone;
            else
                zone = [];
            end
        end
        
        function info = get.TrajectoryInfo(obj)
            if obj.IsConnected
                info = obj.getTrajectoryInfo;
            else
                info = [];
            end
        end
        
        
        function sendWaypoints(obj,Q,T) 
            % blocking function
             % Q Joint Positions [numPoints, numJoints]
            % T Time Vector is [numPoints]            
            [numPoints,~] = size(Q);
            if numPoints ~= length(T)
               error('Number of points is not equal'); 
            end
            
            qCmd = zeros(obj.NumJoints,1);
            fCmd = zeros(obj.NumFingers,1);
            for ii=1:numPoints
                tStart = tic;
                qCmd = Q(ii,1:obj.NumJoints)';
                fCmd = Q(ii,obj.NumJoints+1:end)';
                obj.sendJointAndFingerPositionCommand(qCmd,fCmd);
                if ii == numPoints
                    deltaTime = 0;
                else
                    deltaTime = T(ii+1)-T(ii);
                end
                % Calculate time that has passed
                elapsedTime = toc(tStart);
                
                % If there is still room for waiting then wait 
                if deltaTime > elapsedTime
                    pause(deltaTime-elapsedTime);
                end
            end
        end
        
       
%         function delete(obj)
%             disp('Calling destructor method'); 
%             obj.disconnect();            
%         end
        
    end

    methods (Access = private)
        
        function positions = getPositions(obj)
            stat = false; %#ok<NASGU>
            pos = zeros(obj.NumJoints+obj.NumFingers,1);
            if coder.target('MATLAB')
                [stat,pos,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_JNT_POS),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getJointsPosition',coder.wref(pos));
            end
            if ~stat
                error('Failed to get positions');
            end
            positions = pos;
        end
        
        function qPos = getJointPositions(obj)
            pos = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            pos = obj.getPositions();
            qPos = pos(1:obj.NumJoints);
        end
        
        function fPos = getFingerPositions(obj)
            pos = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            pos = obj.getPositions();
            fPos = pos(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
        end
        
        function velocities = getVelocities(obj)
            stat = false; %#ok<NASGU>
            vel = zeros(obj.NumJoints+obj.NumFingers,1);
            if coder.target('MATLAB')
                [stat,~,vel,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_JNT_VEL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getJointsVelocity',coder.wref(vel));
            end
            if ~stat
                error('Failed to get velocities');
            end
            velocities = vel;
        end
        
        function qVel = getJointVelocities(obj)
            vel = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            vel = obj.getVelocities();
            qVel = vel(1:obj.NumJoints);
        end 
        
        function fVel = getFingerVelocities(obj)
            vel = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            vel = obj.getVelocities();
            fVel = vel(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
        end
        
        function torques = getTorques(obj)
            stat = false; %#ok<NASGU>
            torque = zeros(obj.NumJoints+obj.NumFingers,1);
            if coder.target('MATLAB')
                [stat,~,~,torque,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_JNT_TORQUE),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getJointsTorque',coder.wref(torque));
            end
            if ~stat
                error('Failed to get torques');
            end
            torques = torque;            
        end
        
        function qTorque = getJointTorques(obj)
            torque = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            torque = obj.getTorques();
            qTorque = torque(1:obj.NumJoints);
        end
        
        function fTorque = getFingerTorques(obj)
            torque = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            torque = obj.getTorques();
            fTorque = torque(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);           
        end
        
        
        function temperatures = getTemperatures(obj)
            stat = false; %#ok<NASGU>
            temp = zeros(obj.NumJoints+obj.NumFingers,1);
            if coder.target('MATLAB')
                [stat,~,~,~,temp,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_JNT_TEMP),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getJointsTemperature',coder.wref(temp));
            end
            if ~stat
                error('Failed to get temperatures');
            end
            temperatures = temp;      
        end
         
        function qTemp = getJointTemperatures(obj)
            temp = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            temp = obj.getTemperatures();
            qTemp = temp(1:obj.NumJoints);
        end
        
        function qTemp = getFingerTemperatures(obj)
            temp = zeros(obj.NumJoints+obj.NumFingers,1); %#ok<PREALL>
            temp = obj.getTemperatures();
            qTemp = temp(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
        end
        
        function pee = getEndEffectorPose(obj)
            stat = false; %#ok<NASGU>
            pose = zeros(6,1);
            if coder.target('MATLAB')
                [stat,~,~,~,~,pose,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_EE_POSE),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getEndEffectorPose',coder.wref(pose));
            end
            if ~stat
                error('Failed to get end effector pose');
            end
            pee = pose;  
        end
        
        function fee = getEndEffectorWrench(obj)
            stat = false; %#ok<NASGU>
            wrench = zeros(6,1);
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,wrench,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_EE_WRENCH),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getEndEffectorWrench',coder.wref(wrench));
            end
            if ~stat
                error('Failed to get end effector wrench');
            end
            fee = wrench;  
        end
        
        function dof = getDOF(obj)
            % no input validation 
            stat = false; %#ok<NASGU>
            DOF = 0;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,DOF,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_DOF),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd, obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('GetDOF',coder.wref(DOF));
            end
            if ~stat
                error('Failed to get number of DOF');
            end
            dof = DOF;
        end  
        
        function oee = getEndEffectorOffset(obj)
            stat = false; %#ok<NASGU>
            offset = zeros(4,1);
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,offset,~,~] =  JacoMexInterface(double(MexFunctionIDs.GET_EE_OFFSET),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getEndEffectorOffset',coder.wref(offset));
            end
            if ~stat
                error('Failed to get end effector offset');
            end
            oee = offset;  
        end
        
        function Pzone = getProtectionZone(obj)
            stat = false; %#ok<NASGU>
            zone = 0;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,zone] =  JacoMexInterface(double(MexFunctionIDs.GET_PROTECT_ZONE),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getProtectionZone',coder.wref(zone));
            end
            if ~stat
                error('Failed to get protection zone');
            end
            Pzone = zone; 
        end
        
        function info = getTrajectoryInfo(obj)
            stat = false; %#ok<NASGU>
            Tinfo = 0;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,Tinfo] =  JacoMexInterface(double(MexFunctionIDs.GET_GLOB_TRAJECTORY),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('getTrajectoryInfo',coder.wref(Tinfo));
            end
            if ~stat
                error('Failed to get TrajectoryInfo');
            end
            info = Tinfo;  
        end
        
        
        function sendJointPositionCommandInternal(obj,cmd)
            % no input validation 
            stat = false; %#ok<NASGU>
            obj.JntPosCmd = cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_JNT_POS),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('sendJointPositions',coder.wref(obj.JntPosCmd));
            end
            if ~stat
                error('Failed to send joint position command');
            end
        end
        
        function sendFingerPositionCommandInternal(obj,cmd)
            % no input validation 
            stat = false; %#ok<NASGU>
            obj.FingerPosCmd = cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_FINGER_POS),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('sendFingerPositions',coder.wref(obj.FingerPosCmd));
            end
            if ~stat
                error('Failed to send finger position command');
            end            
        end
        
        function sendJointAndFingerPositionCommandInternal(obj,jntCmd,fCmd)
             % no input validation 
            stat = false; %#ok<NASGU>
            obj.JntPosCmd = jntCmd;
            obj.FingerPosCmd = fCmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_JNT_FING_POS),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('sendJointAndFingerPositions',...
                    coder.wref(obj.JntPosCmd),coder.wref(obj.FingerPosCmd));
            end
            if ~stat
                error('Failed to send joint and finger position command');
            end            
        end  
        
        function sendJointVelocityCommandInternal(obj,cmd)
            % no input validation 
            stat = false; %#ok<NASGU>
            obj.JntVelCmd = cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_JNT_VEL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('sendJointVelocities',coder.wref(obj.JntVelCmd));
            end
            if ~stat
                error('Failed to send joint velocity command');
            end            
        end      
        
        function sendJointTorqueCommandInternal(obj,cmd)
            % no input validation 
            stat = false; %#ok<NASGU>
            obj.JntTorqueCmd = cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_JNT_TORQUE),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('sendJointTorques',coder.wref(obj.JntTorqueCmd));
            end
            if ~stat
                error('Failed to send joint velocity command');
            end               
        end
            
        function sendCartesianPositionCommandInternal(obj,cmd)
            % no input validation 
            stat = false; %#ok<NASGU>
            obj.CartPosCmd = cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_CART_POS),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd, obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('SendCartesianPositions',coder.wref(obj.CartPosCmd));
            end
            if ~stat
                error('Failed to send cartesian position command');
            end               
        end  
        
        function sendCartesianVelocityCommandInternal(obj,cmd)
            % no input validation 
            stat = false; %#ok<NASGU>
            obj.CartVelCmd = cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SEND_CART_VEL),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('SendCartesianVelocity',coder.wref(obj.CartVelCmd));
            end
            if ~stat
                error('Failed to send cartesian velocity command');
            end               
        end  
             
        function setEndEffectorOffsetInternal(obj,Cmd)
            stat = false; %#ok<NASGU>
            obj.OffsetCmd = Cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SET_EE_OFFSET),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('setEndEffectorOffset');
            end
            if ~stat
                error('Failed to set end effector offset');
            end
        end
        
        function setProtectionZoneInternal(obj,Cmd)
            stat = false; %#ok<NASGU>
            obj.ZoneCmd = Cmd;
            if coder.target('MATLAB')
                [stat,~,~,~,~,~,~,~,~,~,~] =  JacoMexInterface(double(MexFunctionIDs.SET_PROTECT_ZONE),...
                    obj.JntPosCmd,obj.JntVelCmd,obj.JntTorqueCmd,obj.FingerPosCmd,obj.CartPosCmd,obj.CartVelCmd,obj.OffsetCmd,obj.ZoneCmd);
            else                
                stat = coder.ceval('setProtectionZone');
            end
            if ~stat
                error('Failed to set protection zone');
            end
        end
               
    end
    
    methods(Access = protected)
        %% Common functions
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            
            % Connect and open library
            obj.connect();
            
            % Set control mode
            if strcmp(obj.ControlMode,'Position')
                obj.setPositionControlMode();
            elseif strcmp(obj.ControlMode,'Velocity')
                obj.setVelocityControlMode();
            elseif strcmp(obj.ControlMode,'Torque')
                obj.setTorqueControlMode();
            else
                error('Selected control mode not defined');
            end                    
               
            % Send to home position
            if obj.GoToHomeAtStartup
                obj.goToHomePosition();
            end
            
            % Do finger calibration
            if obj.CalibrateFingersAtStartup
                obj.calibrateFingers();
            end                        
                
            
        end

        function [JointPos,JointVel,JointTorque,JointTemp,FingerPos,FingerVel,FingerTorque,FingerTemp,Pee,Fee,DOF,Oee, done] = stepImpl(obj,varargin)
            stat = false;
            done = 0;

            % Process inputs 
            if varargin{3} == JacoCommandModes.IDLE
                % do nothing
            elseif varargin{3} == JacoCommandModes.DIRECT_INPUT   % Check the Send Command signal is true                    
                % Send joint commands
                if strcmp(obj.ControlMode,'Position')
                    jntCmd = varargin{1};
                    fCmd = varargin{2};
                    obj.sendJointAndFingerPositionCommandInternal(jntCmd,fCmd)
                elseif strcmp(obj.ControlMode,'Velocity')
                    jntCmd = varargin{1};
                    obj.sendJointVelocityCommandInternal(jntCmd);
                elseif strcmp(obj.ControlMode,'Torque')
                    jntCmd = varargin{1};
                    obj.sendJointTorqueCommandInternal(jntCmd);
                else
                    error('Selected control mode not defined');
                end
            elseif varargin{3} == JacoCommandModes.SEND_POSITION_CMD
                % send command if the command mode input changed
                if varargin{3} ~= obj.CommandModePastValue
                    jntCmd = varargin{1};
                    obj.sendJointPositionCommandInternal(jntCmd);
                end
                
            elseif varargin{3} == JacoCommandModes.SEND_FINGER_CMD
                if varargin{3} ~= obj.CommandModePastValue
                    fCmd = varargin{2};
                    obj.sendFingerPositionCommandInternal(fCmd);
                end
             elseif varargin{3} == JacoCommandModes.SEND_CART_POSITION_CMD
                % send command if the command mode input changed
                if varargin{3} ~= obj.CommandModePastValue
                    cartCmd = varargin{4};
                    % To pass a position matrix
                    col = size(cartCmd);
                    for index=1:col(2)
                        Cmd = cartCmd(:,index);
                        obj.sendCartesianPositionCommandInternal(Cmd);
                    end
                    
                end
            elseif varargin{3} == JacoCommandModes.SEND_CART_VELOCITY_CMD
                 % send command if the command mode input changed
                if varargin{3} ~= obj.CommandModePastValue
                    cartCmd = varargin{4};
                    % To pass a position matrix
                    col = size(cartCmd);
                    for index=1:col(2)
                        Cmd = cartCmd(:,index);
                        for i=1:10
                            obj.sendCartesianVelocityCommandInternal(Cmd);
                        end
                    end
                    
                end
            end
            % Update past command value
            obj.CommandModePastValue = varargin{3};
            if obj.TrajectoryInfo(1) == 0
                % Confirm the move is done
                done = 1;
      
            end

            
            % Generate outputs by reading sensor information               
            % Position
            pos = obj.getPositions();
            JointPos = pos(1:obj.NumJoints);
            FingerPos = pos(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
            
            % Velocity
            vel = obj.getVelocities();
            JointVel = vel(1:obj.NumJoints);
            FingerVel = vel(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
            
            % Torque
            tau = obj.getTorques();
            JointTorque = tau(1:obj.NumJoints);
            FingerTorque = tau(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
            
            % Temperature
            temp = obj.getTemperatures();
            JointTemp = temp(1:obj.NumJoints);
            FingerTemp = temp(obj.NumJoints+1:obj.NumJoints+obj.NumFingers);
            
            % End effector pose
            Pee = obj.getEndEffectorPose();
                                   
            % End effector wrench
            Fee = obj.getEndEffectorWrench();  
            
            % Degree of freedom
            DOF = obj.getDOF();
            
            % Endeffector offset
            Oee = obj.getEndEffectorOffset();
                       

        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            obj.CommandModePastValue = int32(-1);
            %disp('Calling reset method');
        end

        
        function releaseImpl(obj)
            % terminate method
            disp('Calling release method');
            
            % disconnect from library
            obj.disconnect();
        end
        

        
        
        %% Backup/restore functions
        function s = saveObjectImpl(obj)
            % Set properties in structure s to values in object obj

            % Set public properties and states
            s = saveObjectImpl@matlab.System(obj);

            % Set private and protected properties
            %s.myproperty = obj.myproperty;
        end

        function loadObjectImpl(obj,s,wasLocked)
            % Set properties in object obj to values in structure s

            % Set private and protected properties
            % obj.myproperty = s.myproperty; 

            % Set public properties and states
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end

        %% Simulink functions
        function ds = getDiscreteStateImpl(obj)
            % Return structure of properties with DiscreteState attribute
            ds = struct([]);
        end
        
        function flag = isInputSizeLockedImpl(obj,index)
            % Return true if input size is not allowed to change while
            % system is running
            flag = true;
        end
        
        function num = getNumInputsImpl(~)
            num = 4;
        end
        
        function [name,name2,name3,varargout] = getInputNamesImpl(obj)
            % Return input port names for System block
            
            % First input depends on control mode
            if strcmp(obj.ControlMode,'Position')
                varargout{1} = 'jnt_pos_cmd';
            elseif strcmp(obj.ControlMode,'Velocity')
                varargout{1} = 'jnt_vel_cmd';
            elseif strcmp(obj.ControlMode,'Torque')
                varargout{1} = 'jnt_torque_cmd';
            else
                error('Selected control mode not defined');
            end
            
            % Fingers position c,d
            varargout{2} = 'f_cmd';
            
            % Send jnt command enable
            %   0: none, idle
            %   1: direct joint command
            %   2: send position command
            %   3: send torque command
            %   4: send finger command
            varargout{3} = 'cmd_mode';
            
            varargout{4} = 'cart_cmd';
                        

        end
        
        function validateInputsImpl(obj,u,u2,u3,varargin)
%             % Validate joint position, velocity, or torque cmd
%             validateattributes(varargin{1}, {'numeric'},...
%                 {'real', 'nonnan', 'finite', '2d','size',[obj.NumJoints 1]},...
%                 'validateInputs', 'Joint Pos/Vel/Torque cmd');
%           
%             % Validate finger positions cmd
%             validateattributes(varargin{2}, {'numeric'},...
%                 {'real', 'nonnan', 'finite', '2d','size',[obj.NumFingers 1]},...
%                 'validateInputs', 'Finger positions cmd');
%             
%             % Validate enable
%             validateattributes(varargin{3}, {'int32'},...
%                 {'real', 'scalar', 'nonnan', 'finite'},...
%                 'validateInputs', 'Command Modes'); 
%             
%             % Validate cartesian positions cmd
%             validateattributes(varargin{4}, {'numeric'},...
%                 {'real', 'nonnan', 'finite', '2d','size',[obj.CartParam 1]},...
%                 'validateInputs', 'Cartesian pos/Vel cmd');
 
 
        end

        function varargout = isOutputFixedSizeImpl(~)
            varargout{1} = true;
            varargout{2} = true;
            varargout{3} = true;
            varargout{4} = true;
            varargout{5} = true;
            varargout{6} = true;
            varargout{7} = true;
            varargout{8} = true;           
            varargout{9} = true;
            varargout{10} = true;
            varargout{11} = true;
            varargout{12} = true;
            varargout{13} = true;
        end
        
        
        function [o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13] = getOutputSizeImpl(obj)
            % Return size for each output port
            o1 = [obj.NumJoints 1];
            o2 = [obj.NumJoints 1];
            o3 = [obj.NumJoints 1];
            o4 = [obj.NumJoints 1];
            o5 = [JacoComm.NumFingers 1];
            o6 = [JacoComm.NumFingers 1];
            o7 = [JacoComm.NumFingers 1];
            o8 = [JacoComm.NumFingers 1];
            o9 = [JacoComm.CartParam 1];
            o10 = [JacoComm.CartParam 1];
            o11 = [1 1];
            o12 = [4 1];
            o13 = [1 1];

            % Example: inherit size from first input port
            % out = propagatedInputSize(obj,1);
        end
        
        function varargout = isOutputComplexImpl(obj)
            varargout{1} = false;
            varargout{2} = false;
            varargout{3} = false;
            varargout{4} = false;
            varargout{5} = false;
            varargout{6} = false;
            varargout{7} = false;
            varargout{8} = false;           
            varargout{9} = false;
            varargout{10} = false;
            varargout{11} = false;
            varargout{12} = false;
            varargout{13} = false;
        end
        
        function varargout = getOutputDataTypeImpl(obj)
            varargout{1} = 'double';
            varargout{2} = 'double';
            varargout{3} = 'double';
            varargout{4} = 'double';
            varargout{5} = 'double';
            varargout{6} = 'double';
            varargout{7} = 'double';
            varargout{8} = 'double';           
            varargout{9} = 'double';
            varargout{10} = 'double';
            varargout{11} = 'double';
            varargout{12} = 'double';
            varargout{13} = 'double';
        end
        
        function icon = getIconImpl(obj)
            % Define icon for System block
            icon = mfilename('class'); % Use class name
            % icon = 'My System'; % Example: text icon
            % icon = {'My','System'}; % Example: multi-line text icon
            % icon = matlab.system.display.Icon('myicon.jpg'); % Example: image file icon
        end
        

        
    end

    methods(Static, Access = protected)
        %% Simulink customization functions
        function header = getHeaderImpl
            % Define header panel for System block dialog
            header = matlab.system.display.Header(mfilename('class'));
        end

        function group = getPropertyGroupsImpl
            % Define property section(s) for System block dialog
            group = matlab.system.display.Section(mfilename('class'));
        end
    end
end
