function topohelmet(W,p1,v1,p2,v2,p3,v3,p4,v4,p5,v5); 
%
% topohelmet(W, ...);
%
% Plot a topographic map of the values in W on a Neuromag helmet  
% 
% If W = [1:61] a Neuromag 122 helmet is assumed
% If W = [1:102] a Neuromag 306 (VectorView) helmet is assumed.   
%
% The files 'NM122coildef.mat' and 'NM306coildef.mat' defining coil locations
% must be available. 
% Optional Parameters & Values (in any order):  
%
%     'maplimits'        - 'absmax' +/- the absolute-max
%                          'maxmin' scale to data range
%                          [clim1,clim2] user-definined lo/hi
%                          {default = 'maxmin'}
%     'coils'            - 'on','off','numbers'
%                          {default = 'off'}
%
% See also COLORBAR, COLORMAP, ROTATE3D
%
%------------------------------------------------------------------------
% Ole Jensen, Brain Resarch Unit, Low Temperature Laboratory,
% Helsinki University of Technology, 02015 HUT, Finland,
% Report bugs to ojensen@neuro.hut.fi
%------------------------------------------------------------------------

%    Copyright (C) 2000 by Ole Jensen 
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You can find a copy of the GNU General Public License
%    along with this package (4DToolbox); if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


W(find(isnan(W))) = 0; 

nargs = nargin;
   
MAPLIMITS = 'maxmin';

if length(W) == 61
    cname = 'NM122coildef.mat';
elseif length(W) == 102
    cname = 'NM306coildef.mat';
else
    error('W must be 1x61 for Neuromag 122 or 1x102 for Neuromag 306');
end

[fid,mess] = fopen(cname,'r');
if ~isempty(mess)
     tmpstr = strcat('Cannot open the file with coil data   : ' ,cname); 
     error(tmpstr); 
end
fclose(fid);   
load(cname)         

ELECTROD = 'off';
  if ~(round((nargs+1)/2) == (1+nargs)/2)
    error('Incorrect number of inputs');
  end
  for i = 3:2:nargs
    Param = eval(['p',int2str((i-3)/2 +1)]);
    Value = eval(['v',int2str((i-3)/2 +1)]);
    if ~isstr(Param)
      error('Parameter must be a string')
    end
    Param = lower(Param);
    switch lower(Param)
       case 'coils'
         ELECTROD = lower(Value);
       case 'maplimits'
         MAPLIMITS = Value;
    otherwise
      error('Unknown parameter.')
    end
  end
% end

% clf




if size(W,1) > size(W,2)
    W = W';
end



VMc = VM;


for j=1:length(W)
    WC(j) = (W(j)-min(W))/(max(W)-min(W)); 
    WC(j) = W(j);
end
WC = WC';
                       
          
    
patch('Vertices',VM,'Faces',FM','FaceVertexCData',WC,'FaceColor','interp','EdgeColor','none','FaceLighting','flat'  ) 

FM = [];
for j=1:length(BRD)-1
    q1  = BRD(j); 

    tmp = VM(BRD(j),:) - [0.0 -0.0001 0.0001];
    VM = [VM' tmp']';
    q2 = size(VM,1);

    tmp = VM(BRD(j+1),:) - [0.0 -0.0001 0.0001];
    VM = [VM' tmp']';
    q3 = size(VM,1);

    q4  = BRD(j+1); 
    FM = [FM [q1 q2 q3 q4]'];                          
end
patch('Vertices',VM,'Faces',FM','FaceVertexCData',[1 1 1],'FaceColor','flat','EdgeColor',[1 1 1]  ) 

hold on

if isstr(MAPLIMITS)
    if strcmp(MAPLIMITS,'absmax')
      amin = -max(abs(W));
      amax = max(abs(W));
    elseif strcmp(MAPLIMITS,'maxmin')
      amin = min(W);
      amax = max(W);
    end
else
    amin = MAPLIMITS(1);
    amax = MAPLIMITS(2);
end
caxis([amin amax]) 

    
if strcmp(ELECTROD,'on')
    for j=1:length(VMc)
        plot3(VMc(j,1),VMc(j,2),VMc(j,3),'.', ...
            'Color',[1 1 1],'markersize',5);
    end
end
if strcmp(ELECTROD,'numbers')
    for j=1:length(VMc)
        plot3(VMc(j,1),VMc(j,2),VMc(j,3),'.', ...
            'Color',[1 1 1],'markersize',5);
        [TH,PHI,R] = cart2sph(VMc(j,1),VMc(j,2),VMc(j,3));
        [X,Y,Z]    = sph2cart(TH,PHI,R+0.010); 
        text(X,Y,Z,num2str(j), ...
            'Color',[0 0 0]);
    end
end

hold off
axis vis3d off

%colorbar
