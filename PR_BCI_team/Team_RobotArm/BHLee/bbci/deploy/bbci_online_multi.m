%BBCI_ONLINE - creates structure with information from N classifiers and
% calls bbci_apply
%
%Synopsis:
%  [DATA, BBCI, RES]= bbci_calibrate_MI(data_dir, cfy_files, log_classif)
%
%   data_dir : directory where the raw files are
%   cfy_files: classifier filenames. In the format cfyname1;cfyname2;cfyname3;...
%
% 05-2012 Javier Pascual

function [data, bbci]= bbci_online(data_dir, cfy_files, log_classif)

  bbci_startup_communications( '11512', '12345','D:\development\mundus\runtime' )
   
%    try,
         
        if(strcmp(cfy_files,'') == 0),
            files = regexp(cfy_files,';','split');
        
            % FEEDBACK CONTROL SIGNALS
            bbci.feedback.control = 1:length(files);
            
            for i=1:length(files),
                f = char(files(i));
                cfy = load([data_dir f]);
                
                % SIGNAL
                bbci.signal(i).source= 1;
                bbci.signal(i).clab = cfy.signal.clab;        
                if(isfield(cfy.signal, 'proc')),
                    bbci.signal(i).proc = cfy.signal.proc;   
                end;
                
                % FEATURES (1 feature x CTRL)
                bbci.feature(i).signal= i;
                bbci.feature(i).param = {{},{}};
                if(isfield(cfy.feature, 'proc')),
                    bbci.feature(i).proc = cfy.feature.proc;
                end;
                bbci.feature(i).fcn = cfy.feature.fcn;
                bbci.feature(i).ival = cfy.feature.ival;     
                
                % CLASSIFIER 
                bbci.classifier(i).feature = i;        
                bbci.classifier(i).C = cfy.classifier.C;    
                
                bbci.control(i).classifier = i;                  
                bbci.control(i).fcn = @bbci_control_BGUI_ERP;
                bbci.control(i).condition = cfy.control.condition;                   
                bbci.control(i).param = cfy.control.param;
                
                if(isfield(cfy, 'adaptation')),                  
                  bbci.adaptation(i) = cfy.adaptation;
                  bbci.adaptation(i).param= {struct('ival',[1500 4000], 'mrk_start', [231], 'mrk_end', [241])};
                  bbci.adaptation(i).classifier = i;
                  bbci.adaptation(i).log.output= 'screen';
                end;
                
            end;                        
        end;        
        
        bbci.source.acquire_param= {struct('fs', 100)};
        bbci.source.record_signals = 0;
        bbci.feedback.receiver = 'pyff';
        %bbci.log.output = 'screen';        
        bbci.log.folder= data_dir;
        bbci.log.classifier= str2double(log_classif);
        bbci.source.reconnect = 1;
        bbci.quit_condition.marker= 255;


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         bbci.source.acquire_fcn= @bbci_acquire_offline;
%         [cnt, mrk_orig] = eegfile_readBV('D:\data\bbciRaw\VPogt_12_05_04\online_MIVPogt*', 'fs', 100);
%    
%         mrk.fs = mrk_orig.fs;
%         mrk.pos = mrk_orig.pos(3:end);
%         mrk.chan = mrk_orig.pos(3:end);
%         mrk.clock = mrk_orig.pos(3:end);
%         mrk.length = mrk_orig.pos(3:end);
%         mrk.time = mrk_orig.pos(3:end);
%         mrk.type = mrk_orig.pos(3:end);
%         mrk.desc = zeros(1, length(mrk_orig.pos)-2);
%         
%         
%         mrk.desc = [];
%         for i=3:length(mrk_orig.desc),
%             s = mrk_orig.desc{i};    
%             if((length(s) > 1) && (length(s) < 5))
%                 mrk.desc(i) = str2num(s(2:length(s)));
%             else,
%                 mrk.desc(i) = 0;
%             end;
%                 
%         end
%         bbci.source.acquire_param= {cnt, mrk};
%         bbci.log.output= 'screen&file';
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [data, bbci] = bbci_apply(bbci);
%         
%     catch ME
%         msg = ['Exception in bbci_online: ' ME.message];
% 
%         for k=1:length(ME.stack)
%             msg = [msg '\n name=' ME.stack(k).name '; line=' num2str(ME.stack(k).line)]
%         end
% 
%         msgbox(msg);                
%     end; 
    
    data = [];
end
