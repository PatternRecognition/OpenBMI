function [data, bbci]= bbci_online(dir, file1, file2, nClasses, nSeq, record, test, log_classif)

   %bbci_startup_communications( '11512', '12345','D:\development\mundus\runtime' )
    
    try
        
        if(strcmp(file2,'') == 0),
            bbci_mi = load([dir file1]);
            bbci_erp = load([dir file2]);

            bbci = [];

            if(str2num(test) == 1),
                i = chanind(bbci_mi.signal.clab, bbci_erp.signal.clab);
                w = bbci_mi.signal.proc{1}{2};
                w = w(i,:);

                newclab = {};
                for j = 1:length(i),            
                    newclab{j} = bbci_mi.signal.clab{i(j)};
                end    
                bbci_mi.signal.proc{1}{2} = w;

                bbci_mi.signal.clab = newclab;
            end

            bbci.source.record_signals = str2num(record);
            bbci.feedback.control = [1 2];

            % MI signal
            bbci.signal(1).source= 1;
            bbci.signal(1).clab = bbci_mi.signal.clab;        
            bbci.signal(1).proc = bbci_mi.signal.proc;          
            % ERP signal
            bbci.signal(2).source= 1;
            bbci.signal(2).clab = bbci_erp.signal.clab;        

            % MI feature
            bbci.feature(1).signal= 1;
            bbci.feature(1).fcn = bbci_mi.feature.fcn;
            bbci.feature(1).param = {{},{}};
            bbci.feature(1).ival = bbci_mi.feature.ival;                 
            % ERP feature
            bbci.feature(2).signal= 2;
            bbci.feature(2).proc = bbci_erp.feature.proc;
            bbci.feature(2).ival = bbci_erp.feature.ival;         

            % MI classifier
            bbci.classifier(1).feature = 1;        
            bbci.classifier(1).C = bbci_mi.classifier.C;        
            % ERP classifier
            bbci.classifier(2).feature = 2;        
            bbci.classifier(2).C = bbci_erp.classifier.C;


            % MI control
            bbci.control(1).classifier = 1;  
            bbci.control(2).fcn = []; 
            % ERP control
            bbci.control(2).classifier = 2;                
            bbci.control(2).fcn = @bbci_control_BGUI_ERP;   
            bbci.control(2).condition = bbci_erp.control.condition;   
            bbci.control(2).param = {struct('nClasses',str2num(nClasses), 'nSequences', str2num(nSeq))};
        else,
            bbci = load([dir file1]);                        
        
            bbci.settings.nClasses= str2num(nClasses);
            bbci.control.fcn = @bbci_control_BGUI_ERP;               
            % jpg: It is not necessary if the calibration is done with the
            % same number of classes
            bbci.control.condition.marker = [11:49];
            bbci.control.param = {struct('nClasses',str2num(nClasses), 'nSequences', str2num(nSeq))};
        end;
        
        bbci.source.record_signals = str2num(record);
        bbci.feedback.receiver = 'pyff';
        bbci.log.output = 'file';        
        bbci.log.folder= dir;
        bbci.log.classifier= str2num(log_classif);
        bbci.source.reconnect = 1;
        bbci.quit_condition.marker= 255;
        
        [data, bbci] = bbci_apply(bbci);
        
    catch ME
        msg = ['Exception in bbci_online: ' ME.message];

        for k=1:length(ME.stack)
            msg = [msg '\n name=' ME.stack(k).name '; line=' num2str(ME.stack(k).line)]
        end

        msgbox(msg);                
    end; 
    
    data = [];
end
