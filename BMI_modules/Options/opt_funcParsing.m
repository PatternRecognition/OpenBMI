function [out_param str_function in_dat in_param type] = opt_funcParsing( str )
%OPT_FUNCPASING Summary of this function goes here
%   Detailed explanation goes here
%% single input parameter (e.g. SMTfb=func_projection(SMTfb, CSP_W))
%% make a division with "{" or "}" symbol
%% The first parameter we consider it as data, and rest of it is parameter(struct, numeric etc. )
%% Numeric parameter should have [], and without it for the the structure or cell variable
%% "str"로 묶여 있다면 string형, 이외는 변수(cell, structure or numeric)
if isempty(strfind(str, '{'))
    str2=opt_getToken(str, '=');
    str_out=str2{1};
    str_out=strrep(str_out,'[','');
    str_out=strrep(str_out,']','');
    out_param=opt_getToken(str_out, ',');
  
    str_in=str2{2,:}; %input paremters
    str_in2=opt_getToken(str_in,'(');
    str_function=str_in2{1};
    str_param=str_in2{2,:};
    str_param=strrep(str_param, ')', ''); %마지막 부분 ) 제거
    str_param=opt_getToken(str_param, ',');
    in_dat=str_param{1};
    in_param=str_param(2:end);
    for i=1:length(in_param)
        if ~isempty(strfind(in_param{i},'"')) %string
            type{i}='string';
            in_param{i}=strrep(in_param{i},'"','');
        else
            [x, status] = str2num(in_param{i});
            if status
                type{i}='numeric';
                in_param{i}=strrep(in_param{i},'[','');
                in_param{i}=strrep(in_param{i},']','');
                in_param{i}=x;
            else
                  type{i}='variable';                  
             end
        end 
    end
    
else
    %% cell type input paremeter (e.g. CNT=prep_filter(CNT, {"frequency", [7 13]}))
    
    str2=opt_getToken(str, '=');
    str_out=str2{1};
    str_out=strrep(str_out,'[','');
    str_out=strrep(str_out,']','');
    out_param=opt_getToken(str_out,',');
    
    str_in=str2{2,:}; %input paremters
    str_in2=opt_getToken(str_in,'(');
    str_function=str_in2{1};
    str_param=str_in2{2,:};
    
    str_param=opt_getToken(str_param, '{');
    str_param_dat=str_param{1,:};
    in_dat=opt_getToken(str_param_dat, ',');
    str_param_pa=opt_getToken(str_param{2,:},';');
    
    str_param_pa{end}=strrep(str_param_pa{end}, ')', ''); %마지막 부분 ) 제거
    str_param_pa{end}=strrep(str_param_pa{end}, '}', '');
    
    
    for i=1:length(str_param_pa) % ; 로 나누어져있을 경우
        tm=opt_getToken(str_param_pa{i},',');
        for j=1:length(tm)
            if ~isempty(strfind(tm{j},'"')) %string
                tCHAR='string';
                tm{j}=strrep(tm{j},'"','');
                type{i,j}=tCHAR;
                in_param{i,j}=tm{j};
            else
                [x, status] = str2num(tm{j});
                if status
                    tCHAR='numeric';
                    tm{j}=strrep(tm{j},'[','');
                    tm{j}=strrep(tm{j},']','');                   
                    tm{j}=x;
                    type{i,j}=tCHAR;
                    in_param{i,j}=tm{j};
                else
                    if isempty(strfind(tm{j},'"')) % unsigned parameter, the input parameter. CV.var will be used for assigned parameter
                    tCHAR='unassigned_variable';
                    type{i,j}=tCHAR;
                    in_param{i,j}=tm{j};
                    else
                    tCHAR='assigned_variable';  % already assgined variable. 
                    tm{j}=strrep(tm{j},'"','');
                    type{i,j}=tCHAR;
                    in_param{i,j}=tm{j};
                    end
                end
            end            
        end
    end
    
end


