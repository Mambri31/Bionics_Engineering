%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola il Mean Square Error (MSE) degli stimatori ML e MAP 
% del parametro eta di una distribuzione esponenziale e ne fa il grafico al variare di N
% La ddp a priori di Eta è ancora un esponenziale di parametro Lambda
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=10^4;                      % Numero di prove Monte Carlo per ogni N dati
Nv=2.^[1:12]';               % Numero di osservati IID (condizionatamente a Eta)
Lambda=1;                    % Parametro della ddp a priori di eta, esponenziale monolatero                  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=waitbar(0,'Please wait...');
for kk=1:length(Nv)
    waitbar(kk/length(Nv),h);
    N=Nv(kk);
    
    for i=1:M
        U1=rand(1,1);
        Eta=-log(U1)./Lambda;        % Generazione della v.a. Eta esponenziale di parametro Lamba
        U=rand(N,1);
        Y=-log(U)./Eta;             % Generazione degli N campioni IID di una v.a. esponenziale di parametro Eta
        YSM=mean(Y);                % Media campionaria degli osservati
        Eta_ML=1./YSM;              % Stima ML di Eta
        Eta_MAP=1./(YSM+Lambda/N);  % Stima MAP di Eta
        ErrEta_ML(i)=Eta-Eta_ML;    % Errore nella stima ML di Eta
        ErrEta_MAP(i)=Eta-Eta_MAP;  % Errore nella stima MAP di Eta 
    end
    
    MSE_ML(kk)=mean(ErrEta_ML.^2);      % MSE nella stima ML di Eta
    MSE_MAP(kk)=mean(ErrEta_MAP.^2);    % MSE nella stima MAP di Eta
    
end
close(h)

MCRB_Eta=(2/Lambda^2)./Nv;     % Bound di Cramer-Rao modificato (MCRB)


figure (1);
semilogx(Nv,10*log10(MSE_ML),'b','LineWidth',2);grid on;
hold on
semilogx(Nv,10*log10(MSE_MAP),'g','LineWidth',2);grid on;
hold on
semilogx(Nv,10*log10(MCRB_Eta),'r','LineWidth',2);
legend('MSE ML estimate of \eta','MSE MAP estimate of \eta', 'MCRB(\eta)');
title('MSE vs N');xlabel('Data length [N]');ylabel('MSE [dB]')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 