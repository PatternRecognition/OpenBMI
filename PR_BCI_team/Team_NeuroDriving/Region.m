    %% Regional Channel Selection
    % 5 Region
    channel_f=epoch.x(:,1:7,:);
    channel_o=epoch.x(:,28:32,:);
    channel_p=epoch.x(:,23:27,:);
    channel_c=epoch.x(:,[8:11,13:15,18:21],:);
    channel_t=epoch.x(:,[12,16:17,22],:);
    
    % 5 Region + Vertical
    channel_f1=epoch.x(:,[1,3:4],:);
    channel_f2=epoch.x(:,[2,6:7],:);
    channel_f3=epoch.x(:,5,:);
    channel_o1=epoch.x(:,29,:);
    channel_o2=epoch.x(:,31,:);
    channel_o3=epoch.x(:,30,:);
    channel_p1=epoch.x(:,[18:19,23:24,28],:);
    channel_p2=epoch.x(:,[20:21,26:27,32],:);
    channel_p3=epoch.x(:,25,:);
    channel_c1=epoch.x(:,[8:9,13],:);
    channel_c2=epoch.x(:,[10:11,15],:);
    channel_c3=epoch.x(:,14,:);
    channel_t1=epoch.x(:,[12,17],:);
    channel_t2=epoch.x(:,[16,22],:);