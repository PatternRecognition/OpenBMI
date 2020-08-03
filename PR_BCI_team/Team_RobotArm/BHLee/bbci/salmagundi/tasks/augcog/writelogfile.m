function writelogfile(typ,data,bl);

persistent fid;

if isempty(fid),
    fid = 1;
end

switch typ     
    case 0
        if ~isempty(data)
            fid = fopen(data,'w');
            ss = data(16:end-4);
            ye = str2num(ss(1:4));
            mo = str2num(ss(5:6));
            da = str2num(ss(7:8));
            ho = str2num(ss(10:11));
            mi = str2num(ss(12:13));
            se = str2num(ss(14:15));
            fprintf(fid,'\n\nStart writing at %s\n\n',datestr([ye,mo,da,ho,mi,se]));
            fprintf(fid,'Sampling Rate: %i Hz\n\n',bl);
        else
            fid = 1;
        end
    case 1
        fprintf(fid,'\n\nEnd writing at %s\n\n',datestr(now));
        if fid>2
            fclose(fid);
        end 
        fid = 1;
    case 2
        fprintf(fid,'Message: %s\n',data);
    case 3
        fprintf(fid,'Send_udp after %d msec: [%d',bl,data(1));
        for i = 2:length(data)
            fprintf(fid,' %d',data(i));
        end
        fprintf(fid,']\n');
    case 4
        fprintf(fid,'Classifier after %d msec: [%f',bl,data(1));
        for i = 2:length(data)
            fprintf(fid,' %f',data(i));
        end
        fprintf(fid,']\n');
    case 5
        fprintf(fid,'ERROR: \n%s\n',data);
end 

        
    
           