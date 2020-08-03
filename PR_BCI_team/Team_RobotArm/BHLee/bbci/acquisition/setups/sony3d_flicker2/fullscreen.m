function fullscreen(image,device_number)
%FULLSCREEN Display fullscreen true colour images
%   FULLSCREEN(C,N) displays 24bit UINT8 RGB matlab image matrix C on
%   display number N (which ranges from 1 to number of screens). Image
%   matrix C must be the exact resolution of the output screen since no
%   scaling in implemented. If fullscreen is activated on the same display
%   as the MATLAB window, use ALT-TAB to switch back.
%
%   If FULLSCREEN(C,N) is called the second time, the screen will update
%   with the new image.
%
%   Use CLOSESCREEN() to exit fullscreen.
%
%   Requires Matlab 7.x (uses Java Virtual Machine), and has been tested on
%   Linux and Windows platforms.
%
%   Written by Pithawat Vachiramon 18/5/2006


ge = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment();
gds = ge.getScreenDevices();
height = gds(device_number).getDisplayMode().getHeight();
width = gds(device_number).getDisplayMode().getWidth();

if ~isa(image,'uint8')
    error('Image matrix must be of UINT8 type');
elseif ~isequal(size(image,3),3)
    error('Image must be NxMx3 RGB');
elseif ~isequal(size(image,1),height)
    error(['Image must have verticle resolution of ' num2str(height)]);
elseif ~isequal(size(image,2),width)
    error(['Image must have horizontal resolution of ' num2str(width)]);
end


global frame_java;
global icon_java;
global device_number_java;

if ~isequal(device_number_java, device_number)
    try frame_java.dispose(); end
    frame_java = [];
    device_number_java = device_number;
end
    
if ~isequal(class(frame_java), 'javax.swing.JFrame')
    frame_java = javax.swing.JFrame(gds(device_number).getDefaultConfiguration());
    frame_java.setUndecorated(true);
    icon_java = javax.swing.ImageIcon(im2java(image)); 
    label = javax.swing.JLabel(icon_java); 
    frame_java.getContentPane.add(label);
    gds(device_number).setFullScreenWindow(frame_java);
else
    icon_java.setImage(im2java(image));
end
frame_java.pack
frame_java.repaint
frame_java.show