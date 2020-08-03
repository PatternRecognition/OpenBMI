%BBCI_CALIBRATE_ERP - Function that trains a ERP classifier from raw
% calibration data.
%
%Synopsis:
%  [DATA, BBCI, RES]= bbci_calibrate_ERP(data_dir, file, nClasses, nSeqOnline, classif_name, ivalstr, chanstr)
%
%   data_dir : directory where the raw files are
%   file: filename
%   nClasses: number of classes of the paradigm (usually 6)
%   nSeqOnline: number of repeticions in the online mode
%   classif_name: name of the output classifier
%   ivalstr: time ival str. In the format ival11;ival12;ival21;ival22
%   chanstr: channels str. In the format chan1;chan2;...;chanN
%
% 05-2012 Javier Pascual

function [bbci, data, res] = bbci_calibrate_ERP(data_dir, file, nClasses, nSeqOnline, classif_name, ivalstr, chanstr)
    
     try
        res = '';
        data = [];
        bbci= [];

        BC= [];
        BC.fcn= @bbci_calibrate_ERP_Speller;
        BC.settings.nClasses= str2num(nClasses);
    
        if(strcmp(ivalstr,'') == 0),
            parts = regexp(ivalstr,';','split');

            ival = zeros(length(parts)/2,2);
            for i=1:length(parts)/2,
                ival(i,1) = str2num(char(parts(2*i-1)));
                ival(i,2) = str2num(char(parts(2*i)));
   
            end;
            
            BC.settings.cfy_ival = ival;
        end;
        
        BC.folder= data_dir;
        BC.file= file;
        BC.marker_fcn= @mrk_defineClasses;
        BC.marker_param= {{[31:49], [11:29]; 'target', 'nontarget'}};

        BC.log.output = 'file';
        BC.log.folder = [data_dir '/log/'];
        if(strcmp(chanstr,'') == 0),
            chan = regexp(chanstr,';','split');
        else
            chan = {'*'};
        end;
        BC.read_param = {'clab', chan, 'fs', 100}
     
        bbci.calibrate= BC;    
        bbci.calibrate.save.file  = classif_name;
                       
        [bbci, data] = bbci_calibrate(bbci);    
        bbci.control.fcn = @bbci_control_BGUI_ERP;
        bbci.control.param = {struct('kind', 'ERP', 'nClasses',str2num(nClasses), 'nSequences', str2num(nSeqOnline), 'cfy', classif_name)};
        bbci.signal.proc = [];
        
        res = 'Result:';
        aux = sprintf('\n-------------------------------');
        res = [res aux]; 
        aux = sprintf('\nROC-loss: %.1f', data.result.rocloss );
        res = [res aux]; 
    
    catch ME
        msg = ['Exception in bbci_calibrate_ERP: ' ME.message];

        for k=1:length(ME.stack)
            msg = [msg '\n name=' ME.stack(k).name '; line=' num2str(ME.stack(k).line)]
        end

        msgbox(msg);        
    end;
end
