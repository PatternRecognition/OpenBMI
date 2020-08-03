% Example code to determine the spherical and Cartesian coordinates of
% four intermediate 10-20 locations (F3, F4, P3, P4) from a small circle
% given by three known 10-20 locations (F7, Fz, F8)
%
% (published in appendix of Kayser J, Tenke CE, Clin Neurophysiol 2006;117(2):348-368)
%
ThetaPhi = [144.000    0.000;                % F7      a priori known
             90.000   45.000;                % Fz      spherical coordinates
             36.000    0.000] ;              % F8      [theta phi]
ThetaPhiRad = (2 * pi * ThetaPhi) / 360;     % convert degrees to radians
[X,Y,Z] = sph2cart(ThetaPhiRad(:,1), ...     % theta [radians] is in column 1
                   ThetaPhiRad(:,2), ...     % phi [radians] is in column 2
                   1.0 );                    % use unit radius
XYZ = [X Y Z];                               % create Cartesian matrix
F7 = XYZ(1,:);                               % recove point vectors with
FZ = XYZ(2,:);                               % ... Cartesian x,y,z coordinates
F8 = XYZ(3,:);                               % ... for the three electrodes
P = SphericalMidPoint(F7,F8,FZ,F7,FZ,'F3');  % determine F3 as F7-Fz mid point
F3 = P(2,:);                                 % use intersection above x-y plane
F4 = F3 .* [-1  1  1];                       % mirror x-coordinate for F4
P3 = F3 .* [ 1 -1  1];                       % mirror y-coordinate for P3
P4 = F3 .* [-1 -1  1];                       % mirror x-,y-coordinates for P4
XYZext = [XYZ; F3; F4; P3; P4];              % extend XYZ matrix with new sites
[th,ph,ra] = cart2sph(XYZext(:,1), ...       % convert Cartesian xyz coordinates
                      XYZext(:,2), ...       % ... to spherical coordinates
                      XYZext(:,3));          % ... [radians]
ThetaPhiExt = [ [th * 360 / 2 / pi] ...      % convert spherical coordinates from
                [ph * 360 / 2 / pi]];        % ... radians to degrees 
Elab = ['F7';'Fz';'F8'; ...                  % create electrode label array 
        'F3';'F4';'P3';'P4'];
disp(sprintf('%8s  %10s %10s %12s %12s %12s', ...
    'Site','Theta','Phi','X','Y','Z'));
for i = 1:size(Elab,1)                       % table coordinates ...
  disp(sprintf('%8s  %10.3f %10.3f %12.5f %12.5f %12.5f', ...
    char(Elab(i,:)), ...                     % ... electrode label
    ThetaPhiExt(i,1), ThetaPhiExt(i,2), ...  % ... Theta, Phi [°]
    XYZext(i,1),XYZext(i,2),XYZext(i,3)) );  % ... Cartesian x,y,z coordinates 
end;