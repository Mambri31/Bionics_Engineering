%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola il Mean Square Error (MSE) degli stimatori
% ML/MMSE/Pseudo-MMSE di Theta1, Theta2 e ampiezza di una oscillazione cosinusoidale 
% immersa in rumore AWGN e ne fa il grafico al variare di SNRdB
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=10^4;                      % Numero di prove Monte Carlo per ogni SNR
varTheta=10;                 % VarTheta=E{A^2}/4B=varA/2B
f0=1;                        % frequenza analogica in KHertz (KHz)
B=5;                         % Banda filtro passa-basso anti-aliasing in KHertz (KHz)
T= 20;                       % intervallo di osservazione misurato in millisecondi (ms)
Tc=1/(2*B);                  % intervallo di campionamento misurato in millisecondi (ms)
N=round(2*B*T);              % Numero di campioni N=T/Tc=2BT deve essere >>1
F0=f0*Tc;                    % frequenza digitale (adim)                         
SNRdB=[-30:1:10];            % Vettore dei rapporti segnale-rumore in decibel [dB]
SNR=10.^(SNRdB./10);         % Vettore di SNR in scala lineare
varW=varTheta./SNR;          % Vettore delel potenze di rumore AWGN, varW=N0/2                
k=[0:N-1]';                  % Vettore del tempo discreto
h1=cos(2*pi*F0*k);
h2=sin(2*pi*F0*k);
varA=varTheta*2*B;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=waitbar(0,'Please wait...');
for kk=1:length(SNR)
    waitbar(kk/length(SNR),h);
    
    alpha(kk)=1./(1+2/(N*SNR(kk)));
    
    for i=1:M
        
        Theta=sqrt(varTheta)*randn(2,1);
        S=Theta(1)*h1+Theta(2)*h2;           % generazione del vettore segnale aleatorio
        Noise=sqrt(varW(kk))*randn(N,1);     % generazione del vettore Nx1 di rumore
        Y=S+Noise;                           % generazione del vettore Nx1 dei dati osservati 
        Stima_Theta1_ML=2/N*h1'*Y;
        Stima_Theta2_ML=2/N*h2'*Y;
        Stima_Theta1_MMSE=alpha(kk)*Stima_Theta1_ML;
        Stima_Theta2_MMSE=alpha(kk)*Stima_Theta2_ML;
        AML=sqrt(2*B)*sqrt(Stima_Theta1_ML^2+Stima_Theta2_ML^2);
        AMMSE=alpha(kk)*AML;
        MAP=(AMMSE+sqrt(AMMSE^2+4*(1-alpha(kk))*varA))/2;
        
        ErrTheta1ML(i)=Theta(1)-Stima_Theta1_ML;                  % Errore nella stima ML di Theta1
        ErrTheta1MMSE(i)=Theta(1)-Stima_Theta1_MMSE;              % Errore nella stima MMSE di Theta1
        ErrAML(i)=sqrt(2*B)*sqrt(Theta(1)^2+Theta(2)^2)-AML;      % Errore nella stima ML di A
        ErrAMMSE(i)=sqrt(2*B)*sqrt(Theta(1)^2+Theta(2)^2)-AMMSE;  % Errore nella stima pseudo-MMSE di A
        ErrMAP(i)=sqrt(2*B)*sqrt(Theta(1)^2+Theta(2)^2)-MAP;      % Errore nella stima MAP di A
    end
    
    MSE_Theta1ML(kk)=mean(ErrTheta1ML.^2);             % MSE nella stima ML di Theta1
    MSE_Theta1MMSE(kk)=mean(ErrTheta1MMSE.^2);         % MSE nella stima MMSE di Theta1
    MSE_AML(kk)=mean(ErrAML.^2);                       % MSE nella stima ML di A
    MSE_AMMSE(kk)=mean(ErrAMMSE.^2);                   % MSE nella stima pseudo-MMSE di A
    MSE_MAP(kk)=mean(ErrMAP.^2);                       % MSE nella stima MAP di A
end
close(h)

BCRB_Theta1=2*alpha.*varW/N;
BCRB_A=2*B*varTheta./(3/2+SNR*N/2);

figure (1);
plot(SNRdB,10*log10(MSE_Theta1ML),'b','LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(MSE_Theta1MMSE),'g','LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(BCRB_Theta1),'r','LineWidth',2);
legend('MSE ML estimate of \theta_{1}','MSE MMSE estimate of \theta_{1}', 'Bayesian CRB(\theta_{1})');
title('MSE vs SNR');xlabel('SNR [dB]');ylabel('MSE [dB]')

figure (2);
plot(SNRdB,10*log10(MSE_AML),'b','LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(MSE_AMMSE),'g','LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(MSE_MAP),'magenta','LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(BCRB_A),'r','LineWidth',2);
legend('MSE ML estimate of A','MSE Pseudo-MMSE estimate of A','MSE MAP estimate of A','Bayesian CRB(A) (approximate)');
title('MSE vs SNR');xlabel('SNR [dB]');ylabel('MSE [dB]')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 