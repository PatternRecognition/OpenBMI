waitForSync;

interval=600;

loud1=0.4
loud2=0.99

while 1,
  %speaker=floor(rand*6)+1;
  for i=1:6
    test_tone(i,5,0.5,opt)
    waitForSync(interval);
    
  end
end


loud_stages_log=logspace(log10(0.0025),log10(0.4),7)
loud_stages_lin=linspace((0.0025),(0.4),7)


loud_stages = (loud_stages_log*0.4+loud_stages_lin*0.6)

loud_stages = [ 0.0035 0.04358 0.08643 0.1334 0.1899 0.2689 0.4];
waitForSync(interval);

while 1,
  speaker=floor(rand*6)+1;
  for i=1:1
    for ii=1:length(loud_stages)
      test_tone(i,40,loud_stages(ii),opt)
      waitForSync(interval);
    end    
  end
end

%% 
waitForSync;
waitForSync(1000);
for i=1:10
  test_tone(6,40,0.4,opt);
  waitForSync(2000);
end

%%
% thomas d
% calibrated for speaker 1, 5 ms, loudness = 0.5 , 68.8 db in max clock
% mode (fast)
volumes=[ 0.5 0.6 0.5 0.6 0.55 0.55 0 0;... % 5 ms
          0.23 0.25 0.22 0.24 0.23 0.25 0 0;... % 10
          0.15 0.15 0.15 0.15 0.14 0.155 0 0;... % 20
          0.1 0.115 0.1 0.12 0.115 0.115 0 0;... % 40
          0.08 0.085 0.085 0.085 0.085 0.085 0 0;... % 80
          0.07 0.08 0.07 0.085 0.07 0.07 0 0;... % 160
          0.06 0.07 0.07 0.08 0.07 0.07 0 0]; % 300    

        
%%
% thomas r
% calibrated for loud_stages on speaker 1 
%loud_stages = [ 0.0035 0.04358 0.08643 0.1334 0.1899 0.2689 0.4];
% duration: 40 ms

volumes=[ 0.0035 0.0035 0.0035 0.0035 0.0035 0.0035 0 0;...      % 52 db almost impossible to measure
          0.04358 0.04358 0.04358 0.04358 0.04358 0.04358 0 0;...      % 55 db almost impossible to measure
          0.08643 0.08643 0.08643 0.08643 0.08643 0.08643 0 0;...% 60 db
          0.1334 0.1334 0.13 0.13 0.13 0.12 0 0;...              % 63.5 db          
          0.1899 0.1899 0.1899 0.1899 0.1899 0.172 0 0;...       % 67 db
          0.2689 0.2689 0.2689 0.2689 0.27 0.255 0 0;...         % 70.5 bd
          0.4 0.4 0.42 0.42 0.42 0.4 0 0];                       % 74.5 bd    
