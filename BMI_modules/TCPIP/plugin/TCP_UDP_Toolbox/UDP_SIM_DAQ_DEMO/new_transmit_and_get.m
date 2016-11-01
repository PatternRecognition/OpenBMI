function [data_matrix,id_list ] = new_transmit_and_get(REMOTEIP,REMOTEPORT,RECVPORT)
%NEW_TRANSMIT_AND_GET transmits a call and gets the data from all channels
%output is an matrix with the data for all the channels
%
% UDP demo using pnet, by Peter Rydesäter, Bollnäs, Sweden
%

pnet('closeall'); %  Prepare by closing existing pnet connections if not properly closed last time

DATASIZE =   12500;
BYTEORDER = 'network';
%BYTEORDER = 'intel';
TIMEOUT = 1.8;                          % Seconds timeout for receive loop
data_matrix=uint16(zeros(DATASIZE,16)); % For efficiency, prepare empty matrix
id_list    =uint16(zeros(16));          % And prepare this, "small one" to to be consequent

%Setup UDP socket for reciving packets to port RECVPORT
udpsock=pnet('udpsocket',RECVPORT);
pnet(udpsock,'setwritetimeout',1);
pnet(udpsock,'setreadtimeout',0);

%Send request. Use reciving sockets as "platform" to create
% and send packet from also... recycling/saving....;-)
disp 'Send request...'
pnet(udpsock,'printf','XMITCALLANDGET,ALL,END');
pnet(udpsock,'writepacket',REMOTEIP,REMOTEPORT);

%Receiving packets. Do some output and receive the 16 channels
disp 'Receive...'
packnr=0;
start_time=clock();
while packnr<16,
    len=pnet(udpsock,'readpacket');
    if len==DATASIZE*2+2, % Valid sized packet received? => take care
        packnr=packnr+1;
        id_list(packnr)       =pnet(udpsock,'read',1,'uint16',BYTEORDER);
        data_matrix(:,packnr) =pnet(udpsock,'read',DATASIZE,'uint16',BYTEORDER);
        fprintf('PACK %02d len=%06d id=%d\n',packnr,len,id_list(packnr));
    elseif len>0,
        fprintf('PACK BAD size:%d \n',len);
    end
    if etime(clock(),start_time) > TIMEOUT,
        fprintf('TIMEOUT....:-(\n');
        break;
    end
end;
pnet(udpsock,'close');
if packnr==16,
    fprintf('All packs received :-)\n');
end

return;

