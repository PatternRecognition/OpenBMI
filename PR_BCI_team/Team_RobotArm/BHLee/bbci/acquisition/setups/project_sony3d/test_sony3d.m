pause_during_pic = .5;

standard_markers;
resources_list;

%calculate duration
num_pics = numel(file_names);

pnet(tcp_conn, 'printf', 'pic %s', 'black');
pause(pause_during_pic);
pnet(tcp_conn, 'printf', 'pic %s', 'press');
pause(pause_during_pic);
pnet(tcp_conn, 'printf', 'pic %s', 'question');
pause(pause_during_pic);

for j=1:num_pics
    fprintf('pic %d/%d\n',j,num_pics);
    pnet(tcp_conn, 'printf', 'pic %s', file_names{j});
    pause(pause_during_pic);
end

pnet('closeall');