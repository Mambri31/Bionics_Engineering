%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola il Mean Square Error (MSE) degli stimatori
% ML di ampiezza e fase di un oscillazione cosinusoidale immersa in rumore
% AWGN e ne fa il grafico al variare di SNRdB
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=5*10^6;                    % Numero di prove Monte Carlo per ogni SNR
A=3;                         % Ampiezza del segnale
Theta=2*pi/3;                % Fase iniziale in radianti
T=10;                        % Intervallo di osservazione
SNRdB=[-20:2:20];            % Vettore dei rapporti segnale-rumore in decibel [dB]
SNR=10.^(SNRdB./10);         % Vettore di SNR in scala lineare
N0=((A^2)*T)./(2*SNR);       % Densità Spettrale di Potenza monolatera del rumore
varW=N0./2;                  % Vettore delle potenze di rumore
S1=A*sqrt(T/2)*cos(Theta);   % generazione della componente di segnale deterministico - componente 1
S2=-A*sqrt(T/2)*sin(Theta);  % generazione della componente di segnale deterministico - componente 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=waitbar(0,'Please wait...');
for k=1:length(SNR)
    waitbar(k/length(SNR),h); 
    
    W1=sqrt(varW(k))*randn(M,1);        % generazione del vettore di rumore - componente 1
    W2=sqrt(varW(k))*randn(M,1);        % generazione del vettore di rumore - componente 2
    X1=S1+W1;                           % generazione del vettore dati osservati - componente 1
    X2=S2+W2;                           % generazione del vettore dati osservati - componente 2
    Ahat=sqrt(2/T)*sqrt(X1.^2+X2.^2);   % Stima ML di A
    Thetahat=-atan2(X2,X1);             % Stima ML di Theta
    ErrA=Ahat-A;                        % Errore nella stima ML di A
    ErrTheta=Thetahat-Theta;            % Errore nella stima ML di Theta
    MSE_A(k)=mean(ErrA.^2);             % MSE nella stima ML di A
    MSE_Theta(k)=mean(ErrTheta.^2);     % MSE nella stima ML di Theta 
end
close(h)

CRB_A=(A^2)./(2*SNR);
CRB_Theta=1./(2*SNR);

figure (1);
plot(SNRdB,10*log10(MSE_A),'LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(CRB_A),'r','LineWidth',2);
legend('MSE ML Estimate of A ','CRB(A)');
title('MSE vs SNR_{out}');xlabel('SNR_{out} [dB]');ylabel('MSE [dB]')

figure (2);
plot(SNRdB,10*log10(MSE_Theta),'LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(CRB_Theta),'r','LineWidth',2);
legend('MSE ML Estimate of \Theta ','CRB(\Theta)');
title('MSE vs SNR_{out}');xlabel('SNR_{out} [dB]');ylabel('MSE [dB]')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 