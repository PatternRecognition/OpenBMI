%% 절대좌표에 기반한 직교좌표계 상에서의 point by point move 함수
%% 현재는 목표 pose에 위치 도달과 자세 제어를 동시에 수행

function [stat] = moveToCP(obj,desired_pos)

% Input size 체크
if (size(desired_pos)~=6)
    disp('Check the size of input');
    stat=-1;
    return;
end    
    
% Cartesian 공간상에 있는 작업범위를 넘어서는지 체크
limit_x=0.9;
limit_y=0.9;
limit_z=1.1;
current_pos=obj.EndEffectorPose;

if abs(desired_pos(1))>limit_x
    disp('Unable to move');
    stat=-1;
    return;
end    

if abs(desired_pos(2))>limit_y
    disp('Unable to move');
     stat=-1;
end

if (desired_pos(3)>limit_z || desired_pos(3)<0) 
    stat=-1;
    disp('Unable to move');
end


%각도 범위를 초과하지 않도록 조절
%각도의 경우 -pi에서 pi의 범위를 갖도록 함

if abs(desired_pos(4))>pi
   desired_pos(4)=mod(desired_pos(4)+pi,2*pi)-pi;
end    

if abs(desired_pos(5))>limit_y
   desired_pos(5)=mod(desired_pos(5)+pi,2*pi)-pi;
end

if abs(desired_pos(6))>limit_y
   desired_pos(6)=mod(desired_pos(6)+pi,2*pi)-pi;
end
% i값과 속도값을 적절히 조절하는 것이 관건 
% Send cartesian velocity
% i=200 은 1초에 해당
% 앞의 셋은 미터지만 뒤는 라디안. 5ms마다 입력이 들어가므로 절대 0.01을 넘는 값을 인가하지 말것
% 뒤의 셋의 경우 대개 small actuator의 한계값과 관련이 있어 
% 뒤에 xyz가 붙은 경우는 직선 움직임 
% error를 1cm 미만으로 만들도록 함


tollerance_xyz=0.01;
%tollerance_angle=0.3;

error_xyz=sqrt((desired_pos(1)-current_pos(1)).^2+(desired_pos(2)-current_pos(2)).^2+(desired_pos(3)-current_pos(3)).^2);
%error_angle=sqrt((desired_pos(4)-current_pos(4)).^2+(desired_pos(5)-current_pos(5)).^2+(desired_pos(6)-current_pos(6)).^2);
timestep=0;
while error_xyz>tollerance_xyz 
    temp_pos=obj.EndEffectorPose;
    
    %% xyz공간 상의 거리 평가
    error_xyz=sqrt((desired_pos(1)-temp_pos(1)).^2+(desired_pos(2)-temp_pos(2)).^2+(desired_pos(3)-temp_pos(3)).^2);
    CartVel=0.2;
    direction=CartVel*[desired_pos(1)-temp_pos(1),desired_pos(2)-temp_pos(2),desired_pos(3)-temp_pos(3)]/error_xyz;
    
%     if(error_xyz<0.1)
%          CartVel=0.9*error_xyz;
%     end
    
%     error_pose=sqrt(abs(mod(desired_pos(4)-temp_pos(4),2*pi)-2*pi).^2+abs(mod(desired_pos(5)-temp_pos(5),2*pi)-2*pi).^2+abs(mod(desired_pos(6)-temp_pos(6),2*pi)-2*pi).^2);
%     % xyz공간 상의 각도 평가.
%     AngleVel=0.1;
%     pose=AngleVel*[desired_pos(4)-temp_pos(4),desired_pos(5)-temp_pos(5),desired_pos(6)-temp_pos(6)]/error_pose;
%     
%     if(error_pose<1)
%          AngleVel=0.1*error_pose;
%     end
%         
%    CartVelCmd = [direction(1);direction(2);direction(3);pose(1);pose(2);pose(3)];
    CartVelCmd = [direction(1);direction(2);direction(3);0;0;0];
    sendCartesianVelocityCommand(obj,CartVelCmd);
    
    timestep=timestep+1;
    if(timestep>2000)
        break;
    end    
end



stat=0;
end

