function monitorUDPport(port, ival),
clc;

handle = get_data_udp(port);
while 1,
    dat = get_data_udp(handle, ival, 0);
    if ~isempty(dat) && ~isnan(dat(5)),
%          fprintf('CLS output: %f\n', dat(5));
        fprintf('%i, %6.4f, %i\n', dat(3), dat(5), dat(6));
    end
end
end