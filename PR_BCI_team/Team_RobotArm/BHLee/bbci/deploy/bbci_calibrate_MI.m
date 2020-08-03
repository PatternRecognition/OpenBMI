%BBCI_CALIBRATE_MI - Function that trains a RLDA MI classifier from raw
% calibration data.
%
%Synopsis:
%  [DATA, BBCI, RES]= bbci_calibrate_MI(data_dir, file, classif_name, chanstr, bandstr, ivalstr, class)
%
%   data_dir : directory where the raw files are
%   file: filename
%   classif_name: name of the output classifier
%   chanstr: channels str. In the format chan1;chan2;...;chanN
%   bandstr: time ival str. In the format band11;band12;band21;band22
%   ivalstr: time ival str. In the format ival11;ival12;ival21;ival22
%   class:
%          'auto': chooses the best combination (l vs r, l vs f or f vs r)
%          'class1;class2': trains with data from class1 and class2 
%
% 05-2012 Javier Pascual

function [bbci, data, res] = bbci_calibrate_MI(data_dir, file, classif_name, chanstr, bandstr, ivalstr, class)
    
    try
        res = '';
        data = [];
        bbci= [];
       
        BC= [];
        BC.fcn= @bbci_calibrate_csp;

        BC.folder= data_dir;
        BC.file= file;
        BC.read_param= {'fs',100};
        BC.marker_fcn= @mrk_defineClasses;
        
        if(strcmp(class,'left')),
            BC.marker_param= {{[1],[2 3 4]; 'left', 'rest'}};
        elseif(strcmp(class,'right')),
            BC.marker_param= {{[2],[1 3 4]; 'right', 'rest'}};            
        elseif(strcmp(class,'foot')),
            BC.marker_param= {{[3],[1 2 4]; 'foot', 'rest'}};            
        elseif(strcmp(class,'left;right')),
            BC.marker_param= {{1 2; 'left', 'right'}};    
        elseif(strcmp(class,'left;foot')),
            BC.marker_param= {{1 3; 'left', 'foot'}};    
        elseif(strcmp(class,'foot;right')),
             BC.marker_param= {{3 2; 'foot', 'right'}};    
        elseif(strcmp(class,'idle')),
             BC.marker_param= {{[1 2 3], [4]; 'MI', 'IDLE'}};                 
        else,
            BC.marker_param= {{1 2 3; 'left', 'right', 'foot'}};            
        end;
        
        BC.log.output = 'file';
        BC.log.folder = [data_dir '/log/'];
      
        if(strcmp(bandstr,'') == 0),
            band = zeros(1,2);
            parts = regexp(bandstr,';','split');
                     
            band(1) = str2num(char(parts(1)));
            band(2) = str2num(char(parts(2)));
   
            BC.settings.band = band;
        end;
        
        if(strcmp(ivalstr,'') == 0),
            ival = zeros(1,2);
            parts = regexp(ivalstr,';','split');
                     
            ival(1) = str2num(char(parts(1)));
            ival(2) = str2num(char(parts(2)));
   
            BC.settings.ival = ival;
        end;
        
        if(strcmp(chanstr,'') == 0),
            chan = regexp(chanstr,';','split');
        else
            chan = {'*'};
        end;
        
        BC.read_param = {'clab', chan, 'fs', 100}
        
        bbci.calibrate= BC;    
        bbci.calibrate.settings.selival_opt.laplace_require_neighborhood = 0;                  
        bbci.calibrate.settings.selband_opt.laplace_require_neighborhood = 0;                  
        bbci.calibrate.settings.selival_opt.max_ival = [500 4300];                  
        bbci.calibrate.settings.laplace_require_neighborhood = 0;
        bbci.calibrate.save.file = classif_name;
        [bbci, data] = bbci_calibrate(bbci);    
        
        res = 'Results:';
        aux = sprintf('\n-------------------------------');
        res = [res aux]; 
        for i = 1:length(data.all_results),
            aux = sprintf('\n<%s vs %s>: %.2f +- %.2f', ...
                            data.all_results(i).classes{:}, data.result.class_selection_loss(1,i)*100, data.result.class_selection_loss(2,i)*100);
            res = [res aux];                    
        end;
        
        aux = sprintf('\n-------------------------------\n', data.result.classes{:});
        res = [res aux]; 
        aux = sprintf('\nSelected classes: %s vs %s\n', data.result.classes{:});
        res = [res aux]
                        
        bbci.control.fcn = [];
        bbci.control.param = {struct('kind', 'MI', 'cfy', classif_name)};
        bbci.control.condition = [];        
        bbci.feature.proc = [];
        bbci.feature.param = {{},{}};
        
        
        bbci.adaptation.active= 1;
        bbci.adaptation.fcn= @bbci_adaptation_pmean;
        bbci.adaptation.param= {struct('ival',[1000 4000], 'mrk_start', [231], 'mrk_end', [241])};
        bbci.adaptation.filename= [data_dir '/bbci_classifier_pmean'];
        bbci.adaptation.log.output= 'screen&file';
        
    catch ME
        msg = ['Exception in bbci_calibrate_MI: ' ME.message];

        for k=1:length(ME.stack)
            msg = [msg '\n name=' ME.stack(k).name '; line=' num2str(ME.stack(k).line)];
        end

        msgbox(msg);        
    end;
end
