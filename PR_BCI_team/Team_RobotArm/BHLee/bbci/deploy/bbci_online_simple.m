function [data, bbci]= bbci_online_simple(file1, file2, nSeq, record,test)

   addpath(genpath('D:\development\bbci_online_new2\toolbox'))
   addpath(genpath('D:\development\bbci_online_new2\online_new'))
   addpath(genpath('D:\development\bbci_online_new2\deploy'))
   addpath(genpath('D:\development\bbci_online_new2\import\tcp_udp_ip'))
   bbci_startup_communications( '11512', '12345' )
   
    try
        bbci = load(file1);
        bbci_erp = load(file2);
        
              
        if(test),
            i = chanind(bbci.signal.clab, bbci_erp.signal.clab);
            w = bbci.signal.proc{1}{2};
            w = w(i,:);

            newclab = {};
            for j = 1:length(i),            
                newclab{j} = bbci.signal.clab{i(j)};
            end    
            bbci.signal.proc{1}{2} = w;

            bbci.signal.clab = newclab;
        end
        
        bbci.source.record_signals = str2num(record);
        bbci.feedback.receiver = 'pyff';
        bbci.log.output = 'file';        
        bbci.source.reconnect = 1;
        
        
        [data, bbci] = bbci_apply(bbci);
        
    catch ME
        msg = ['Exception in bbci_online: ' ME.message];

        for k=1:length(ME.stack)
            msg = [msg '\n name=' ME.stack(k).name '; line=' num2str(ME.stack(k).line)]
        end

        msgbox(msg);                
    end; 
end
