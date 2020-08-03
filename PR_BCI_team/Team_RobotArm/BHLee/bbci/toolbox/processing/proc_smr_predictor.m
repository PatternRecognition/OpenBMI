function [indi_smr, yhat, noise] = proc_smr_predictor(spec, chan, fitmode, frange)
 % [indi_smr, yhat, noise] = proc_smr_predictor(spec, chan, fitmode,frange)
 %
 % calculate the SMR predictor in one channel
 %
 % IN   spec     - data structure of Fourier-transformed EEG data
 %      chan     - number of channel for which the SMR predictor shall be calculated
 %      fitmode  - fitmode = 'area'    => Predictor value = area between noise floor and spectrum
 %                 fitmode = 'maxdiff' => Predictor value = max. difference
 %                 between noise floor and spectrum
 %      frange   - frequency range in which the predictor is evaluated 
 %
 % OUT  indi_smr - SMR predictor value (= NaN if fit fails)
 %      yhat     - fitted model function for the spectrum 
 %      noise    - fitted 1/f noise spectrum
 %
 % TD, dickhaus@cs.tu-berlin.de

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
 % fit function y = k(1) + k(2)/ t^lambda + k(3) *phi1(t|mu(1), sigma(1)) + k(4) * phi2(t|mu(2), sigma(2)) %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 start = [1.0, 10.0, 20.0, 1.0, 1.0];
 tt = spec.t;
 yy = spec.x(:, chan);
 %myopts = optimset('MaxFunEvals', 100000, 'MaxIter', 10000);
 %estimated_params = fminsearch(@(x)smr_fitfun2(x,tt,yy),start,myopts);
 estimated_params = fminsearch(@(x)smr_fitfun2(x,tt,yy),start);
 lambda=estimated_params(1);
 mu(1) = estimated_params(2);
 mu(2) = estimated_params(3);
 sigma(1) = estimated_params(4);
 sigma(2) = estimated_params(5);
 A = ones(length(tt),4);
 A(:,2) = 1 ./ (tt.^lambda);
 A(:,3) = normpdf(tt, mu(1), sigma(1));
 A(:,4) = normpdf(tt, mu(2), sigma(2));
 k = A\yy;
 yhat = k(1) + k(2)./tt.^lambda + k(3) * normpdf(tt, mu(1), sigma(1)) + k(4) * normpdf(tt, mu(2), sigma(2));
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % fit function y = k(1)/ t^lambda + k(2) at points ttt1, ttt2, ttt3 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 start = 1.0;
 ttt1 = find(~isnan(spec.x(:, chan)), 1, 'first');
    
 %find end of spectrum
 ende = find(~isnan(spec.x(:, chan)), 1, 'last');
    
 % model first peak
 ttt2 = find(spec.x(ttt1:ende-1, chan) < spec.x(ttt1+1:ende, chan), 1, 'first');
 if(isempty(ttt2))
  flag=1;
 else
  flag=0;  
 end  
    
 %take minimum of spectrum as third argument point
 [hase, indix] = min(spec.x(:, chan));
 ttt3 = min(indix);
    
 if((ttt2 == ttt3) && (flag == 0)) %relative and absolute minimum coincide
   ttt3 = ende;
 end;
    
 if(flag == 1)  %failed to find a minimum in between ttt1 and ttt3
   ttt3 = ende;
   ttt2 = floor(size(spec.t, 2) / 4);
 end;
    
 % apply minimization algorithms
 tt = [spec.t(ttt1), spec.t(ttt2), spec.t(ttt3)];
 yy = spec.x([ttt1, ttt2, ttt3], chan);
 if(yy(3) <= yy(2))
   estimated_lambda = fminsearch(@(x)smr_fitfun1(x,tt,yy),start);
   A = ones(length(tt),2);
   A(:,2) = 1 ./ (tt.^estimated_lambda);
   k = A\yy;
   noise = k(1) + k(2)./spec.t.^estimated_lambda;
   
   if(isempty(frange))
     stuetz=find(noise <= yhat);
   else
     stuetz = intersect(find(spec.t >= frange(1)), find(spec.t <= frange(2)));
     stuetz = intersect(stuetz, find(noise <= yhat));
   end
   if(length(stuetz) < 2) 
      indi_smr = NaN;
   else  
      %% Area in peaks
      if strcmp(fitmode, 'area'),
         addi = trapz(spec.t(stuetz), yhat(stuetz)-noise(stuetz));   
         if((abs(addi) > length(spec.x(:, chan))*max(spec.x(:, chan))) || (addi < 0) || isnan(addi))
           indi_smr = NaN;
         else
           indi_smr = addi;
         end;   
      end; 

      %% largest peak
      if strcmp(fitmode, 'maxdiff'),
         addi = max(yhat(stuetz)-noise(stuetz));
         if((addi > (max(spec.x(:, chan)) - min(spec.x(:, chan)))) || (addi < 0) || isnan(addi))
           indi_smr = NaN;
         else
           indi_smr =  addi;
         end;
      end;
    end     
 else
  indi_smr = NaN;
  noise = NaN*zeros(1, length(spec.t));
 end; 
% end function 'calculate_smr_predictor' % 

