%% Aus setup_sony3d.m

global tcp_conn;
addpath([BCI_DIR 'import/tcp_udp_ip']);

pnet('closeall');
tcp_conn = pnet('tcpconnect', 'sony3d.ml.tu-berlin.de', 12345);
if tcp_conn<0
  error('Couldn''t connect to server.');
end
pnet(tcp_conn, 'setwritetimeout', 1)
pnet(tcp_conn, 'printf', 'msg waiting_for_experiment\n');
pause();



%% View images
conditions = {'Cube2D', 'Cube3D_near', 'Cube3D_medium','Cube3D_far','Scene_2D','Scene_near', 'Scene_medium','Scene_far'};
c = 1;
 
fmt_str = '%s\\%s_sbs_vert%i';

pause_during_pic = 4.5;
pause_before_pic = 1;
pause_after_pic  = 1;
num_repetitions = 10;
shifts=0:4:24;
user_entry=zeros(num_repetitions,numel(shifts))

for i = 1:num_repetitions
  for j=randperm(numel(shifts))
    vert=shifts(j)
    
%     if vert==0
%       pnet(tcp_conn, 'printf', 'msg reference\n');
%     end
%     pause(0.5);

    pnet(tcp_conn, 'printf', 'msg black\n');
    pause(pause_before_pic);

    filename = sprintf(fmt_str, conditions{c}, conditions{c}, vert);
    fprintf('%s\n',filename)
    pnet(tcp_conn, 'printf', 'pic %s\n',filename);

    pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg black\n');
    pause(pause_after_pic);
    pnet(tcp_conn, 'printf', 'msg question\n');
    user_entry(i,j) = quality_input();
  end
end

figure(c)
x=shifts;
y = mean(user_entry);
e = std(user_entry)/sqrt(num_repetitions);
errorbar(x,y,e)
title(conditions{c})
ylim([0.5 3.5]);set(gca, 'XTick', shifts)
ylabel('Rating (+/-SEM)')
xlabel('Shift in pixel')
       

pnet(tcp_conn, 'printf', 'msg black\n');pause(5);
pnet(tcp_conn, 'close');
fprintf('Done!\n');




    

%% ACHTUNG MANCHE BILDER LAUFEN UEBER DIE GANZE BILDSCHIRMBREIT 16:9 ANDERE
% NUR 4:3!!! OBWOHL BILDGROESSE EIGENTLICH GLEICH





