%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola analiticamente l'MSE dei 3 stimatori di S
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varA=1;                   % Varianza del segnale
etaA=10;                  % Valor medio del segnale

% Per la figura MSE vs. Gamma (con varA fissa e varW che varia)

N=64;                     % Numero di campioni osservati
GammadB=[-30:2:10];       % Vettore dei rapporti varianza segnale - varianza rumore in decibel [dB]
Gamma=10.^(GammadB./10);  % Vettore di Gamma in scala lineare
varW=varA./Gamma;         % Vettore delle potenze di rumore

% Per la figura MSE vs. N

Nvect=[1:2:1000]';          % Vettore del numero di campioni osservati
Gamma0dB=-10;               % Rapporto varianza segnale - varianza rumore in decibel [dB] 
Gamma0=10.^(Gamma0dB./10);  % Gamma in scala lineare 
varW0=varA./Gamma0;         % Potenze di rumore 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figura MSE vs. N

Len=length(Nvect);
MSE_ap=varA*ones(Len,1);
MSE_ML=varW0./Nvect;
alpha=(Gamma0*Nvect)./(1+Gamma0*Nvect);
MSE_MMSE=alpha.*MSE_ML;

figure (1);
loglog(Nvect,MSE_ap,'r','Linewidth', 2);grid on;
hold on
loglog(Nvect,MSE_ML,'g','Linewidth', 2);
loglog(Nvect,MSE_MMSE,'b','Linewidth', 2);
legend('MSE A-Priori','MSE ML', 'MSE MMSE');
title('MSE vs N');xlabel('Data size (N)');ylabel('Mean Square Error(MSE)')

% Figura MSE vs. Gamma (con varS fissa e varw che varia) 

Len=length(GammadB);
MSE_ap=varA*ones(Len,1);
MSE_ML=varW./N;
alpha=(N*Gamma)./(1+N*Gamma);
MSE_MMSE=alpha.*MSE_ML;

figure (2);
semilogy(GammadB,MSE_ap,'r', 'Linewidth', 2);grid on;
hold on
semilogy(GammadB,MSE_ML,'g','Linewidth', 2);
semilogy(GammadB,MSE_MMSE,'b','Linewidth', 2);
legend('MSE A-Priori','MSE ML', 'MSE MMSE');
title('MSE vs \gamma');xlabel('\gamma [dB]');ylabel('Mean Square Error(MSE)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
 