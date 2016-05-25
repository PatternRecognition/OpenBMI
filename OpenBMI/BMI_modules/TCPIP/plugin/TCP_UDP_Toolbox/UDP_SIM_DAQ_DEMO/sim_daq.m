function sim_daq(REMOTEIP,REMOTEPORT,RECVPORT)
%SIM_DAQ -- Simulate DAQ system
%
% UDP demo using pnet, by Peter Rydesäter, Bollnäs, Sweden
%

pnet('closeall'); %  Prepare by closing existing pnet connections if not properly closed last time

BYTEORDER = 'network';
%BYTEORDER = 'intel';

DATASIZE = 12500;

sin_wave_data=uint16( (sin( (1:DATASIZE).* (2*pi*10/DATASIZE) )*1000)+1000 );

%Setup UDP socket for reciving packets to port RECVPORT
udpsock=pnet('udpsocket',RECVPORT);
pnet(udpsock,'setwritetimeout',1);
pnet(udpsock,'setreadtimeout',1);

%Receiving packets. Do some output and receive the 16 channels
disp 'SIM_DAQ Waits for request...'
while 1,
    len=pnet(udpsock,'readpacket');
    if len>0, % Packet contain data => Check message
        msg   =pnet(udpsock,'read',100,'char');
        fprintf('MSG: %s\n SEND respons: ',msg);
        tic();  % Start measure send time...
        for packnr=1:16,
            pnet(udpsock,'write',uint16(packnr-1));                 % Send Packnr-1 as id
            pnet(udpsock,'write',sin_wave_data+uint16(packnr*200)); % Send sin wave + DC level as data
            pnet(udpsock,'writepacket',REMOTEIP,REMOTEPORT);
            fprintf(' %d,',packnr);
            pause(0.03); % You need a very small pause between each packet.
                         % else will packets be lost. 
        end
        sec=toc();
        fprintf('END %g s, %3.2f MBytes/s \n\n',sec, 16*(DATASIZE+1)*2/sec/1024/1024);
    end
end;
