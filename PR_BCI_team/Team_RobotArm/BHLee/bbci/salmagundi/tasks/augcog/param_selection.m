setup_augcog;

%fil = 13;
feature = {'bandPowerbyvariance'};
task = {'*auditory','*calc', '*visual', '*carfollow'};
ival = {5000};
ival_apply = {30000};
    
CL = {{'base*','low*'},{'base*','high*'},{'low*','high*'}};
freq = {[3 20],[3 7],[7 10],[7 14],[10 14]};
spatial = {{{'cmr'}}};
params = {{{}}};

modell = {'LDA',struct('classy','RLDA','param',struct('value',[0,0.0001,0.001,0.01,0.1,0.4,0.7],'index',2),'msDepth',3)};
modellname = {'LDA','RLDA'};

channels = {{'not','E*','M*'},{'not','E*','M*','Fp*','F7,8','TP9,10'},{'F#','FC#','C#','P#','CP#','O#'},{'F#','P#','O#'}}; 

global AUGCOG_VALIDATION
AUGCOG_VALIDATION = 'LRO';

for fi = 1:length(fil)
  clear traces;
  
  for ta = 1:length(task)
    blk = getAugCogBlocks(augcog(fil(fi)).file);
    blk = blk_selectBlocks(blk,task{ta});
    if ~isempty(blk.y)
      [cnt,mrk] = readBlocks(blk);
      
      for fe = 1:length(feature)
      
        for iv = 1:length(ival)
          for fr = 1:length(freq)
            for sp = 1:length(spatial{fe})
              for pa = 1:length(params{fe})
                for ch = 1:length(channels)
                  for iva = 1:length(ival_apply)
                    
                    fprintf('Choosing subject: %s\n',augcog(fil(fi)).file);
                    fprintf('Choosing task: %s\n',task{ta});
                    fprintf('Choosing feature: %s\n',feature{fe});
                    fprintf('Choosing step-width: %i and Window-length: %i\n',ival{iv});
                    fprintf('Choosing Frequency-band: [%i %i]\n',freq{fr});
                    fprintf('Choosing Spatial Filter: ');
                    fprintf('%s ',spatial{fe}{sp}{:});
                    fprintf('\n');
                    fprintf('Choosing Parameter: ');
                    if length(params{fe}{pa})>0 & isnumeric(params{fe}{pa}{1})
                      fprintf('%f ',params{fe}{pa}{:});
                    else
                      fprintf('%s ',params{fe}{pa}{:});
                    end
                    fprintf('\n');
                    fprintf('Choosing Channels: ');
                    fprintf('%s ',channels{ch}{:});
                    fprintf('Ival apply: %f',ival_apply{iva});
                    fprintf('\n');
                  
                    for cl = 1:length(CL)
                      

                      for model = 1:length(modell)
                        mode = modell{model};
                        modename = modellname{model};
                        
                        traces = get_ongoing_classification(cnt,mrk_selectClasses(mrk,CL{cl}{:}),feature{fe},ival{iv},[1000 ival_apply{iva}],mode,freq{fr},spatial{fe}{sp},params{fe}{pa}{:},channels{ch});
                        
                        str = sprintf('%s_',channels{ch}{:});
                        str = sprintf('/home/tensor/dornhege/calcData/param_select/%s_%s%s_%s_%s_%i_%i_%i_%s_%i_%i_%s_%s',augcog(fil(fi)).file,CL{cl}{1}(1:end-1),CL{cl}{2}(1:end-1),task{ta}(2:end),feature{fe},ival{iv},freq{fr}(1),freq{fr}(2),spatial{fe}{sp}{1},pa,ival_apply{iva},modename,str(1:end-1));
                        
                        parameter = {augcog(fil(fi)).file,CL{cl},task{ta},feature{fe},ival{iv},freq{fr},spatial{fe}{sp},params{fe}{pa},ival_apply{iva},mode,channels{ch}};
                      
                        save(str,'traces','parameter');
                      end
                      
                    end
                    
                  
                  end
                  
                end
              end
            end
          end
        end
      end
    end
  end
end
