d = tcpip('localhost', 3000, 'NetworkRole', 'Client');
set(d, 'OutputBufferSize', 1024); % Set size of receiving buffer, if needed. 

%Trying to open a connection to the server.
while(1)
    try 
        fopen(d);
        break;
    catch 
        fprintf('%s \n','Cant find Server');
    end
end
connectionSend = d;

% fwrite(d,data.control(ic).packet{2});
fwrite(d,1);


%server nirs
t_n = tcpip('localhost', 3000, 'NetworkRole', 'Server');
set(t_n , 'InputBufferSize', 3001);
% Open connection to the client.
fopen(t_n);
fprintf('%s \n','Client Connected');
connectionServer = t_n;
set(connectionServer,'Timeout',.1);
a(i)=fread(t_n,1,'int32')



