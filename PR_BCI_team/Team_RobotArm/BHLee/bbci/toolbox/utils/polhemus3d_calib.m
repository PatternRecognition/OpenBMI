function [state,dat]=polhemus3d_calib(varargin)
% [state,dat]=polhemus3d_calib(savefile,headerfile)
%
% Read the 3d coordinates of the 3d-tracker device.
% 
% IN:  savefile  - the name of the file where to store the values.
%      headerfile- the name of the file where to find a clab struct.
% OUT: state     - success (1) or failure (0).
%      dat       - 3d-coordinates of electrodes.
%
% globz EEG_RAW_DIR

% kraulem 08/05
global EEG_RAW_DIR
if length(varargin)>0
    savefile = varargin{1};
else
    savefile = [];
end
if length(varargin)>1
    headerfile = varargin{2};
    clab = readGenericHeader(headerfile);
    clab = {clab{chanind(clab,'not','E*','xE*')}};
    clab = {'Periauricular L','Periauricular R','Nasion',clab{:}};
else
    clab = {'Periauricular L','Periauricular R','Nasion','Cz','Fz','Pz'};
end

coord_thresh = 0.5/2.54;
ser=sconfig;

% get the anatomical coordinate system landmarks:
coord_ok = false;
while ~coord_ok
    for ii = 1:3
        disp([num2str(ii) '.      ' clab{ii} ' : ']);
        dat(:,ii) = readEntry(ser);
    end
    % the hyperplane through nasion, orthogonal to periL-periR:
    n_vec = dat(:,1)-dat(:,2);
    n_vec = n_vec/norm(n_vec);
    offset = sum(n_vec.*dat(:,3));
    % distances from hyperplane:
    dL = abs(sum(n_vec.*dat(:,1))-offset);
    dR = abs(sum(n_vec.*dat(:,2))-offset);
    if abs(dL-dR)<coord_thresh
        % Everything's fine.
        coord_ok = true;
        ii=4;
    else    
        % large deviance.
        disp(sprintf('Head landmarks are not symmetric: dL = %2.1f, dR = %2.1f.',dL,dR));
    end 
end
% Read in entry by entry from the serial port:
while ii<=length(clab)
    disp([num2str(ii) '.      ' clab{ii} ' : ']);
    dat(:,ii) = readEntry(ser);
    disp(sprintf('\b\b\b done.'));
    if mod(ii,5)==0|ii==length(clab)
        % it's time to ask for permission to proceed.
        messg = 'Repeat (enter number)/ Exit (x)/ Proceed (Enter)? ';
        s = input(messg,'s');
        for jj=1:(length(messg)+length(s)+2)
            disp(sprintf('\b'));
        end
        switch s
            case {'X','x'}
                % exit
                state = 0;
                return;
            case ''
                % proceed
                ii = ii+1;
                %disp(d(:,ii)');
            otherwise    
                % try to find numerical content
                ii = str2num(s);
        end  
    else
        % Just proceed without confirmation.
        ii = ii+1;    
    end 
    if ~isempty(savefile)
        save([EEG_RAW_DIR savefile],'dat','clab');
        disp([savefile ' written.']);
    else
        disp('No savefile given.');
        keyboard
    end
end 

% Transform data to standard positions:
dat = rotate_refsys(dat,dat(:,1:3));

% Show what we've just read:
plot3(dat(1,:),dat(2,:),dat(3,:),'x');
hold on;
plot3(dat(1,1),dat(2,1),dat(3,1),'ro');
plot3(dat(1,2),dat(2,2),dat(3,2),'go');
plot3(dat(1,3),dat(2,3),dat(3,3),'mo');

axis equal;
cameramenu
% say goodbye:
fclose(ser);
delete(ser);
clear ser;
if ~isempty(savefile)
    save([EEG_RAW_DIR savefile],'dat','clab');
    disp([savefile ' written.']);
else
    disp('No savefile given.');
    keyboard
end 

state = 1;
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dat = readEntry(ser)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read one position.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count=0;
% empty the stack and wait for the next sensor buttonpress.
while count==0|data1(1)~=1
    [data1,count,msg]=fscanf(ser,'%g %g');
end
[data2,count,msg]=fscanf(ser,'%g %g');
% subtract the reference sensor:
dat = data1(2:4)-data2(2:4);
% rotate into standard coordinate system:
dat = rotate_xyz(dat,data2(5:7));
return

function out = sconfig
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FastTrack initalisation
%    
% by Yakob
% 15-Jun-2005
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

port = 'COM4';
% see if a serial object is still blocking this port:
% If so, close the connection.
obj = instrfind('Type','serial');
for ii=1:length(obj)
    if strcmp(get(obj(ii),'Port'),'COM4')&strcmp(get(obj(ii),'Status'),'open')
        fclose(obj(ii));
        clear obj(ii) ;
    end 
end 

s = serial('COM4');
set(s, 'Name', 'SCOM4');
set(s, 'BaudRate', 115200);
set(s, 'DataBits', 8);
set(s, 'FlowControl', 'none');
set(s, 'InputBufferSize', 128);
set(s, 'OutputBufferSize', 1024);
set(s, 'Parity', 'none');
set(s, 'ReadAsyncMode', 'continuous');
set(s, 'StopBits', 1);
set(s, 'Terminator','CR');
%set(s, 'Timeout', 10);
fopen(s);
% set the movement thresholds to zero:
fprintf(s,'I 1,0');
fprintf(s,'I 2,0');
% don't use the other channels:
fprintf(s,'I 3,10000');
fprintf(s,'I 4,10000');
set(s, 'Terminator','CR/LF');
% set the format for each sensor:
fprintf(s,'O1 2,1');
fprintf(s,'O2 2,4,1');
%fprintf(s,'O3 52,1');
%fprintf(s,'O4 52,1');
if nargout > 0 
    out = [s]; 
end
return

function dat=rotate_xyz(dat,r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns the rotated point.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rz = r(1); %azimuth
ry = r(2); % elevation
rx = r(3); % roll

% get the radiant values for the angles.
rx = rx*pi/180;
ry = ry*pi/180;
rz = rz*pi/180;

R_x = [1 0 0; 0 cos(rx) sin(rx); 0 -sin(rx) cos(rx)];
R_y = [cos(ry) 0 -sin(ry); 0 1 0; sin(ry) 0 cos(ry)];
R_z = [cos(rz) sin(rz) 0; -sin(rz) cos(rz) 0; 0 0 1];


R_xyz = R_x*R_y*R_z;
dat = R_xyz*dat;
return


function dat=rotate_refsys(dat,ref)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns the rotated points in terms of the reference system.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% project the third point on the connecting line between the 
% other two. The resulting point will be the new origin.
L = ref(:,1);
R = ref(:,2);
N = ref(:,3);
normal = (R-L)/norm(R-L);
orig = L+(N-L)'*normal*normal;
% translate to new origin.
ref = ref-repmat(orig,[1 3]);

% Rotate first refpoint into xy-plane:
ry = acos(sign(ref(3,1))*ref(1,1)/sqrt(ref(1,1)^2+ref(3,1)^2));
R_y = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
ref = R_y*ref;

% Rotate first refpoint into xz-plane:
rz = acos(sign(ref(2,1))*ref(1,1)/sqrt(ref(1,1)^2+ref(2,1)^2));
R_z = [cos(rz) sin(rz) 0; -sin(rz) cos(rz) 0; 0 0 1];
ref = R_z*ref;

% Rotate third refpoint into xy-plane:
rx = acos(sign(ref(3,3))*ref(2,3)/sqrt(ref(3,3)^2+ref(2,3)^2));
R_x = [1 0 0; 0 cos(rx) sin(rx); 0 -sin(rx) cos(rx)];
ref = R_x*ref;

% If necessary, rotate by pi around y-axis and z-axis:
R_yz = [sign(ref(1,1)) 0 0; ...
	0 sign(ref(2,3)) 0; ...
	0 0 sign(ref(1,1))*sign(ref(2,3))];
ref = R_yz*ref;

% Now perform the transformation on the data:
dat = dat-repmat(orig,[1 size(dat,2)]);
dat = R_yz*R_x*R_z*R_y*dat;
return