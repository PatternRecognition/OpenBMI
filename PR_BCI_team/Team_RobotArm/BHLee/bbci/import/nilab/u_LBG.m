%----------------------------------------------------------------
%         calculate OD/Intensity and concentration changes
%----------------------------------------------------------------
% fun:     u_LBG / u_popLBG
%
%
%----------------------------------
%            function
%----------------------------------
%    [inter-optode distance]          : between S&D in cm
%    [N first datapoints for baseline]: for calculation of OD/Intensity use
%             all or N datapoints [e.g. 50] to extract an average 'baseline' value
%             -'all': takes mean of all datapoints
%             - N datapoints: takes the first N datapoints
%    [select from spectrum] : select source of absorbtion-coefficients/epsilon
%       ->the absorbtion coefficients are plottet below (epsilon of oxy-Hb / deoxy-Hb)
%    NOTE     : if not indicated before : select specific wavelengths before selecting a spectrum
%    [DPF]    : select DP-factor (essenpreis)
%
%=======================================================================
%                                                      Paul, BNIC 2007
%=======================================================================


function ni=u_LBG(ni)



chk=1;

if isempty(ni.file)
    chk=0;
    disp('no data loaded');
end
%#####################################
if chk==1
    %----------------------------------
    %     make fields if not exist
    %----------------------------------

    if isfield(ni,'LBodistance')==0   ;         ni.LBodistance   =2.5       ;end
    if isfield(ni,'LBbaseline')==0    ;         ni.LBbaseline    ='all'     ;end
    %_______________________________________________________DPF [highWL lowWL]
    if isfield(ni,'LBdpf')==0         ;         ni.LBdpf         =[0.92   1.1 ]*6.5    ;end


end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if chk==1

    %----------------------------------
    %         calc concentration changes
    %----------------------------------

    dat            =  ni.dat;
    e              =  ni.LBepsilon;
    DPF            =  ni.LBdpf;

    baseline       =  ni.LBbaseline;
    Loptoddistance =  ni.LBodistance;
    %----------------------------------
    %      1. split data to lower and higher wavelength
    %----------------------------------

    s1=size(dat,1);
    s2=size(dat,2);

    dat=dat+eps;

    dat_highWL=    dat(:,1     :  s2/2);%higher Wavelength
    dat_loweWL=    dat(:,s2/2+1:  end);%lower Wavelength

    flag= isfield(ni,'LBmode');
    
    % #########################################
    if flag==6% mode number6

%         dat=dat-repmat(mean(dat,1),[size(dat,1),1]);

        dat_highWL=    dat(:,1     :  s2/2);%higher Wavelength
        dat_loweWL=    dat(:,s2/2+1:  end);%lower Wavelength

         Att_highWL= real(-log10( (dat_highWL  )./ ...
                ( repmat(mean(dat_highWL,1), [s1,1]))   ))    ;

            Att_loweWL= real(-log10( (dat_loweWL  )./ ...
                ( repmat(mean(dat_loweWL,1), [s1,1]))   ))    ;


        wavelength=ni.wl        ;

%         flag_DPF_correction=1;
%         if flag_DPF_correction == 1
%             load Charite_DPF_correction
%             index_DPF1 = find(DPF_correction(:,1) == wavelength(1));
%             index_DPF2 = find(DPF_correction(:,1) == wavelength(2));
%         end

        load COPE_e_coef; %% load the extinction coefficient file
        index_wav1 = find(e_coef(:,1) == wavelength(1));
        index_wav2 = find(e_coef(:,1) == wavelength(2));
        wav1_ecoef = e_coef(index_wav1,2:3);
        wav2_ecoef = e_coef(index_wav2,2:3);
%         if flag_DPF_correction == 1
%             wav1_ecoef = wav1_ecoef .* DPF_correction(index_DPF1, 2);
%             wav2_ecoef = wav2_ecoef .* DPF_correction(index_DPF2, 2);
%         end
        DPF=4;
        dist=Loptoddistance;
        tot_ecoef = [wav1_ecoef; wav2_ecoef];
        tot_ecoef = tot_ecoef .* DPF .* dist;
        coefMat = pinv(tot_ecoef);

        for  i = 1:size(dat_highWL,2)
            dum=[];
            dum = (coefMat * [dat_highWL(:,i)';dat_loweWL(:,i)'])';

            cc_oxy(:,i)=dum(:,1);
            cc_deo(:,i)=dum(:,2);

            %     oxyData(:,iterCol) = oxydxy(:,1);
            %     dxyData(:,iterCol) = oxydxy(:,2);
        end;



        % oxyData = zeros(rawdataLength,channelNum);
        % dxyData = zeros(rawdataLength,channelNum);
        % for(iterCol = 1:channelNum)
        %     oxydxy = (coefMat * [deltaData(:,2*(iterCol-1)+1)';deltaData(:,2*iterCol)'])';
        %     oxyData(:,iterCol) = oxydxy(:,1);
        %     dxyData(:,iterCol) = oxydxy(:,2);
        % end;


        % #########################################




    else
        %----------------------------------
        %       2.OD
        %----------------------------------
        if strcmp(baseline,'all')==1 %take mean of whole timecourse
            Att_highWL= real(-log10( (dat_highWL  )./ ...
                ( repmat(mean(dat_highWL,1), [s1,1]))   ))    ;

            Att_loweWL= real(-log10( (dat_loweWL  )./ ...
                ( repmat(mean(dat_loweWL,1), [s1,1]))   ))    ;

        else %take first datapoints
            baseline=str2num(baseline);
            Att_highWL= real(-log10( (dat_highWL  )./ ...
                ( repmat(mean(dat_highWL(1: baseline,:),1), [s1,1]))   ))    ;

            Att_loweWL= real(-log10( (dat_loweWL  )./ ...
                ( repmat(mean(dat_loweWL(1: baseline,:),1), [s1,1]))   ))    ;
        end

        A(:,1)=reshape(Att_highWL,s1*s2/2   ,1);
        A(:,2)=reshape(Att_loweWL,s1*s2/2   ,1);

        %----------------------------------
        %       3.cc
        %----------------------------------
        % e=...looks like this
        %               oxy-Hb         deoxy-Hb
        % higherWL: 830 | e: 0.974       0.693
        % lowerWL : 690 | e: 0.35         2.1

        %         e=ones(2)%e=flipud( (e))

        if 1
            e=e/10;

            e2=   e.* [DPF' DPF']  .*  Loptoddistance;
            c= ( inv(e2)*A'  )' ;

            cc_oxy       =reshape(c(:,1),s1,s2/2); %in mmol/l
            cc_deo       =reshape(c(:,2),s1,s2/2); %in mmol/l

        end

%         if 0
%             tot_ecoef =  [0.9291    0.7987
%                 0.3123    2.1382]
%             coefMat = pinv(tot_ecoef);
% 
%             for  i = 1:size(dat_highWL,2)
%                 dum=[];
%                 %       dum = (coefMat * [Att_highWL(:,i)';Att_loweWL(:,i)'])';
%                 dum = [Att_highWL(:,i)';Att_loweWL(:,i)']'*coefMat;
%                 cc_oxy(:,i)=dum(:,1);
%                 cc_deo(:,i)=dum(:,2);
% 
%                 %     oxyData(:,iterCol) = oxydxy(:,1);
%                 %     dxyData(:,iterCol) = oxydxy(:,2);
%             end;
% 
% 
%         end




        % Backward Abacktrafo= c/inv(e2');back to OD

    end
    %----------------------------------
    %         tags
    %----------------------------------


    ni.dat=[cc_oxy cc_deo];

    if isfield(ni,'functions')==0
        ni.functions{1,1}=[    'ni=u_LBG(ni);  % calc concentration changes'];
    else
        ni.functions{end+1,1}=[ 'ni=u_LBG(ni); % calc concentration changes'];
    end

    if isfield(ni,'info')==0
        ni.info{1,1}=    ['**  concentration changes calculated' ];
    else
        ni.info{end+1,1}=['**  concentration changes calculated' ];
    end


end





